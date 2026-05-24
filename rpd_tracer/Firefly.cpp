#include "Firefly.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace rpdtracer {
namespace firefly {

// -----------------------------------------------------------------------------
// Measurement Helpers
// -----------------------------------------------------------------------------
MeasurementAnalysis read_latest_measurements(const char* role,
                                             std::size_t windowSize,
                                             const Measurement* measurements,
                                             int count) {
    MeasurementAnalysis analysis;
    if (role == nullptr || measurements == nullptr || count <= 0) {
        return analysis;
    }

    analysis.samples.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        if (std::strncmp(measurements[index].node, role, sizeof(measurements[index].node)) == 0) {
            analysis.samples.push_back(measurements[index]);
        }
    }

    if (analysis.samples.empty()) {
        return analysis;
    }

    std::sort(analysis.samples.begin(), analysis.samples.end(), [](const Measurement& lhs, const Measurement& rhs) {
        return lhs.timestampNs < rhs.timestampNs;
    });

    if (analysis.samples.size() > windowSize) {
        analysis.samples.erase(analysis.samples.begin(), analysis.samples.end() - static_cast<long>(windowSize));
    }

    if (analysis.samples.size() < 2U) {
        analysis.averageOffset = analysis.samples.front().offset;
        analysis.driftRate = 0.0;
        return analysis;
    }

    // Median filter: reject outlier probes before regression
    {
        std::vector<int64_t> offsets;
        offsets.reserve(analysis.samples.size());
        for (const auto& s : analysis.samples)
            offsets.push_back(s.offset);
        std::sort(offsets.begin(), offsets.end());

        const size_t n = offsets.size();
        const int64_t q1 = offsets[n / 4];
        const int64_t q3 = offsets[3 * n / 4];
        const int64_t iqr = q3 - q1;
        const int64_t lo = q1 - 3 * iqr;
        const int64_t hi = q3 + 3 * iqr;

        analysis.samples.erase(
            std::remove_if(analysis.samples.begin(), analysis.samples.end(),
                [lo, hi](const Measurement& m) { return m.offset < lo || m.offset > hi; }),
            analysis.samples.end());

        if (analysis.samples.size() < 2U) {
            analysis.averageOffset = analysis.samples.empty() ? 0 : analysis.samples.front().offset;
            analysis.driftRate = 0.0;
            return analysis;
        }
    }

    double sumX = 0.0;
    int64_t sumY = 0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    const double referenceTime = static_cast<double>(analysis.samples.front().timestampNs);

    for (const Measurement& measurement : analysis.samples) {
        const double timeDelta = static_cast<double>(measurement.timestampNs) - referenceTime;
        int64_t offsetValue = measurement.offset;
        sumX += timeDelta;
        sumY += offsetValue;
        sumXY += timeDelta * offsetValue;
        sumX2 += timeDelta * timeDelta;
    }

    const double sampleCount = static_cast<double>(analysis.samples.size());
    const double denominator = sampleCount * sumX2 - (sumX * sumX);
    if (std::fabs(denominator) < 1e-10) {
        analysis.averageOffset = sumY / sampleCount;
        analysis.driftRate = 0.0;
        return analysis;
    }

    const double slope = (sampleCount * sumXY - sumX * sumY) / denominator;

    // Use median offset instead of mean for robustness
    std::vector<int64_t> filteredOffsets;
    filteredOffsets.reserve(analysis.samples.size());
    for (const auto& s : analysis.samples)
        filteredOffsets.push_back(s.offset);
    std::sort(filteredOffsets.begin(), filteredOffsets.end());
    analysis.averageOffset = filteredOffsets[filteredOffsets.size() / 2];
    analysis.driftRate = slope;
    return analysis;
}

// -----------------------------------------------------------------------------
// Networking Utilities
// -----------------------------------------------------------------------------
bool enable_sw_timestamps(int sockFd) {
    const int timestampOptions = SOF_TIMESTAMPING_TX_SOFTWARE |
                                 SOF_TIMESTAMPING_RX_SOFTWARE |
                                 SOF_TIMESTAMPING_SOFTWARE |
                                 SOF_TIMESTAMPING_OPT_TSONLY;

    if (setsockopt(sockFd, SOL_SOCKET, SO_TIMESTAMPING, &timestampOptions, sizeof(timestampOptions)) < 0) {
//        std::fprintf(stderr, "ChronoSync: [enable_sw_timestamps] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
        return false;
    }
    return true;
}

bool send_probe(int sockFd, sockaddr_in* peerAddress, timestamp_t* sendTime, int probeId) {
    if (peerAddress == nullptr || sendTime == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - Invalid arguments\n");
        return false;
    }

    char messageBuffer[NETWORK_BUFFER_SIZE] = {};
    if (std::snprintf(messageBuffer, sizeof(messageBuffer), "Probe %d", probeId) < 0) {
//        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - snprintf failed\n");
        return false;
    }

    if (sendto(sockFd,
               messageBuffer,
               std::strlen(messageBuffer),
               0,
               reinterpret_cast<sockaddr*>(peerAddress),
               sizeof(*peerAddress)) < 0) {
//        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - sendto failed (errno: %s)\n", std::strerror(errno));
        return false;
    }

    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC, &timeSpec) == 0) {
        *sendTime = timespec_to_ns(timeSpec);
    } else {
//        std::fprintf(stderr, "ChronoSync: [send_probe] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
        *sendTime = 0;
    }
    return true;
}

void receive_probe(int sockFd, timestamp_t* receiveTime, int expectedProbeId) {
    if (receiveTime == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [receive_probe] CRITICAL - Invalid receiveTime pointer\n");
        return;
    }

    char messageBuffer[NETWORK_BUFFER_SIZE] = {};
    sockaddr_in senderAddress{};
    msghdr messageHeader{};
    iovec bufferVector[1];
    char controlBuffer[NETWORK_BUFFER_SIZE] = {};

    bufferVector[0].iov_base = messageBuffer;
    bufferVector[0].iov_len = sizeof(messageBuffer);

    messageHeader.msg_iov = bufferVector;
    messageHeader.msg_iovlen = 1;
    messageHeader.msg_name = &senderAddress;
    messageHeader.msg_namelen = sizeof(senderAddress);
    messageHeader.msg_control = controlBuffer;
    messageHeader.msg_controllen = sizeof(controlBuffer);

    const ssize_t receivedBytes = recvmsg(sockFd, &messageHeader, 0);
    if (receivedBytes < 0) {
        *receiveTime = 0;
        return;
    }

    if (static_cast<std::size_t>(receivedBytes) >= sizeof(messageBuffer)) {
        *receiveTime = 0;
        return;
    }

    messageBuffer[receivedBytes] = '\0';
    int receivedProbeId = 0;
    if (std::sscanf(messageBuffer, "Probe %d", &receivedProbeId) != 1 || receivedProbeId != expectedProbeId) {
        *receiveTime = 0;
        return;
    }

    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC, &timeSpec) == 0) {
        *receiveTime = timespec_to_ns(timeSpec);
    } else {
//        std::fprintf(stderr, "ChronoSync: [receive_probe] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
        *receiveTime = 0;
    }
}

int tcp_handshake(const char* role, const char* peerIp, int tcpPort) {
    if (role == nullptr || peerIp == nullptr)
        return -1;

    std::fprintf(stderr, "ChronoSync: connecting to %s (role %s, port %d)\n", peerIp, role, tcpPort);
    const bool isNodeA = (std::strcmp(role, "A") == 0);
    int tcpSocketFd = -1;
    sockaddr_in tcpAddress{};
    tcpAddress.sin_family = AF_INET;
    tcpAddress.sin_port = htons(tcpPort);

    if (isNodeA) {
        tcpSocketFd = socket(AF_INET, SOCK_STREAM, 0);
        if (tcpSocketFd < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
            return -1;
        }

        if (inet_aton(peerIp, &tcpAddress.sin_addr) == 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - Invalid peer IP\n");
            close(tcpSocketFd);
            return -1;
        }

        bool connected = false;
        for (int attempt = 0; attempt < CONNECTION_RETRY_LIMIT; ++attempt) {
            if (connect(tcpSocketFd, reinterpret_cast<sockaddr*>(&tcpAddress), sizeof(tcpAddress)) == 0) {
                connected = true;
                break;
            }
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] WARNING - connect attempt %d failed (errno: %s)\n",
//                         attempt + 1,
//                         std::strerror(errno));
            sleep(CONNECTION_RETRY_DELAY_SEC);
        }

        if (!connected) {
            std::fprintf(stderr, "ChronoSync: failed to connect to %s:%d after %d attempts\n", peerIp, tcpPort, CONNECTION_RETRY_LIMIT);
            close(tcpSocketFd);
            return -1;
        }

        constexpr char HANDSHAKE_READY[] = "READY";
        if (send(tcpSocketFd, HANDSHAKE_READY, sizeof(HANDSHAKE_READY), 0) < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - send failed (errno: %s)\n", std::strerror(errno));
            close(tcpSocketFd);
            return -1;
        }

        char ackBuffer[NETWORK_BUFFER_SIZE] = {};
        const ssize_t received = recv(tcpSocketFd, ackBuffer, sizeof(ackBuffer) - 1, 0);
        if (received <= 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] WARNING - recv failed (errno: %s)\n", std::strerror(errno));
        } else {
            ackBuffer[received] = '\0';
        }
    } else {
        int listenSocketFd = socket(AF_INET, SOCK_STREAM, 0);
        if (listenSocketFd < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
            return -1;
        }

        int reuseAddress = 1;
        if (setsockopt(listenSocketFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddress, sizeof(reuseAddress)) < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            return -1;
        }

        tcpAddress.sin_addr.s_addr = INADDR_ANY;
        if (bind(listenSocketFd, reinterpret_cast<sockaddr*>(&tcpAddress), sizeof(tcpAddress)) < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - bind failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            return -1;
        }

        if (listen(listenSocketFd, 1) < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - listen failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            return -1;
        }

        sockaddr_in clientAddress{};
        socklen_t clientLength = sizeof(clientAddress);
        tcpSocketFd = accept(listenSocketFd, reinterpret_cast<sockaddr*>(&clientAddress), &clientLength);
        close(listenSocketFd);

        if (tcpSocketFd < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - accept failed (errno: %s)\n", std::strerror(errno));
            return -1;
        }

        char handshakeBuffer[NETWORK_BUFFER_SIZE] = {};
        const ssize_t received = recv(tcpSocketFd, handshakeBuffer, sizeof(handshakeBuffer) - 1, 0);
        if (received > 0) {
            handshakeBuffer[received] = '\0';
        }

        constexpr char HANDSHAKE_ACK[] = "ACK";
        if (send(tcpSocketFd, HANDSHAKE_ACK, sizeof(HANDSHAKE_ACK), 0) < 0) {
//            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - send failed (errno: %s)\n", std::strerror(errno));
            close(tcpSocketFd);
            return -1;
        }
    }

    std::fprintf(stderr, "ChronoSync: connected to %s (role %s, port %d)\n", peerIp, role, tcpPort);
    return tcpSocketFd;
}

// -----------------------------------------------------------------------------
// Clock Helpers
// -----------------------------------------------------------------------------

timespec hw_now() {
    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC, &timeSpec) != 0) {
//        std::fprintf(stderr, "ChronoSync: [hw_now] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
    }
    return timeSpec;
}

void svc_update_ns(SvcState* state, int64_t offsetNs, double drift) {
    if (state == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [svc_update_ns] CRITICAL - Null svc state\n");
        return;
    }

    const unsigned sequence = state->sequence.fetch_add(1U, std::memory_order_acq_rel);
    state->referenceTime = hw_now();
    state->offset = offsetNs;
    state->drift = drift;
    state->sequence.store(sequence + 2U, std::memory_order_release);
}

// -----------------------------------------------------------------------------
// Firefly
// -----------------------------------------------------------------------------
void firefly_run(const char* role, MeasurementBuffer& buffer) {
    if (role == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [firefly_run] CRITICAL - Invalid arguments\n");
        return;
    }

    int localCount = 0;
    {
        std::lock_guard<std::mutex> lock(buffer.mutex);
        localCount = buffer.count;
    }

    const MeasurementAnalysis analysis = read_latest_measurements(role,
                                                                  REGRESSION_WINDOW_SIZE,
                                                                  buffer.measurements.data(),
                                                                  localCount);

    // Real clock drift is < 100 ppm; anything larger indicates bad data
    double drift = analysis.driftRate;
    if (std::fabs(drift) > 500e-6) {
//        std::fprintf(stderr, "ChronoSync: [firefly_run] WARNING - Drift %.9e exceeds 500 ppm, clamping to 0\n", drift);
        drift = 0.0;
    }

    svc_update_ns(firefly::g_pSvcState, static_cast<int64_t>(analysis.averageOffset * CONSENSUS_ALPHA), drift * CONSENSUS_ALPHA);
//    std::fprintf(stderr,
//                 "ChronoSync: [firefly_run] INFO - Role %s, samples=%zu, offset=%lld, drift=%.9e\n",
//                 role,
//                 analysis.samples.size(),
//                 static_cast<long long>(firefly::g_pSvcState->offset),
//                 firefly::g_pSvcState->drift);
}

// -----------------------------------------------------------------------------
// UDP Node
// -----------------------------------------------------------------------------
static void record_measurement(MeasurementBuffer& buffer, const Measurement& measurement) {
    std::lock_guard<std::mutex> lock(buffer.mutex);
    if (buffer.count < static_cast<int>(buffer.measurements.size())) {
        buffer.measurements[buffer.count] = measurement;
        ++buffer.count;
    }
}

void run_node(const char* role,
              const char* ipA,
              const char* ipB,
              int udpPort,
              MeasurementBuffer& buffer,
              bool& done) {
    if (role == nullptr || ipA == nullptr || ipB == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - Invalid arguments\n");
        return;
    }

    const bool isNodeA = (std::strcmp(role, "A") == 0);
    int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd < 0) {
//        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
        return;
    }

    timeval timeoutValue{};
    timeoutValue.tv_sec = SOCKET_TIMEOUT_MSEC / 1000;
    timeoutValue.tv_usec = (SOCKET_TIMEOUT_MSEC % 1000) * 1000;
    if (setsockopt(socketFd, SOL_SOCKET, SO_RCVTIMEO, &timeoutValue, sizeof(timeoutValue)) < 0) {
//        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
        close(socketFd);
        return;
    }

    if (!enable_sw_timestamps(socketFd)) {
        close(socketFd);
        return;
    }

    sockaddr_in localAddress{};
    sockaddr_in peerAddress{};
    localAddress.sin_family = AF_INET;
    peerAddress.sin_family = AF_INET;
    localAddress.sin_port = htons(udpPort);
    peerAddress.sin_port = htons(udpPort);

    const char* localIp = isNodeA ? ipA : ipB;
    const char* peerIp = isNodeA ? ipB : ipA;

    if (inet_aton(localIp, &localAddress.sin_addr) == 0 || inet_aton(peerIp, &peerAddress.sin_addr) == 0) {
//        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - Invalid IP address\n");
        close(socketFd);
        return;
    }

    if (bind(socketFd, reinterpret_cast<sockaddr*>(&localAddress), sizeof(localAddress)) < 0) {
//        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - bind failed (errno: %s)\n", std::strerror(errno));
        close(socketFd);
        return;
    }

    int tcpSocketFd = tcp_handshake(role, peerIp, udpPort);
    if (tcpSocketFd < 0) {
        close(socketFd);
        return;
    }
    close(tcpSocketFd);

    char messageBuffer[NETWORK_BUFFER_SIZE];
    int skippedIterations = 0;
    int probeIndex = 0;

    while (done == false) {
        ++probeIndex;
        timestamp_t sendTimeA = 0;
        timestamp_t recvTimeA = 0;
        timestamp_t sendTimeB = 0;
        timestamp_t recvTimeB = 0;

        if (isNodeA) {
            if (!send_probe(socketFd, &peerAddress, &sendTimeA, probeIndex)) {
                ++skippedIterations;
                continue;
            }
            receive_probe(socketFd, &recvTimeA, probeIndex);
            if (recvTimeA == 0) {
                ++skippedIterations;
                continue;
            }

            if (std::snprintf(messageBuffer, sizeof(messageBuffer), "%lld %lld",
                              static_cast<long long>(sendTimeA),
                              static_cast<long long>(recvTimeA)) < 0) {
                continue;
            }

            if (sendto(socketFd, messageBuffer, std::strlen(messageBuffer), 0,
                       reinterpret_cast<sockaddr*>(&peerAddress), sizeof(peerAddress)) < 0) {
                ++skippedIterations;
                continue;
            }

            sockaddr_in senderAddress{};
            socklen_t senderLength = sizeof(senderAddress);
            const ssize_t receivedBytes = recvfrom(socketFd,
                                                   messageBuffer,
                                                   sizeof(messageBuffer) - 1,
                                                   0,
                                                   reinterpret_cast<sockaddr*>(&senderAddress),
                                                   &senderLength);
            if (receivedBytes < 0) {
                ++skippedIterations;
                continue;
            }

            messageBuffer[receivedBytes] = '\0';
            if (std::sscanf(messageBuffer, "%lld %lld",
                            reinterpret_cast<long long*>(&sendTimeB),
                            reinterpret_cast<long long*>(&recvTimeB)) != 2) {
                ++skippedIterations;
                continue;
            }

            const timestamp_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
            const int64_t offset            = (recvTimeB - sendTimeA) - (roundTripTime / 2);

            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            Measurement measurement{};
            measurement.timestampNs = timespec_to_ns(now);
            measurement.sendTimeA = sendTimeA;
            measurement.recvTimeA = recvTimeA;
            measurement.sendTimeB = sendTimeB;
            measurement.recvTimeB = recvTimeB;
            measurement.roundTripTime = roundTripTime;
            measurement.offset = offset;
            measurement.udpPort = udpPort;
            measurement.node[0] = 'A';
            measurement.node[1] = '\0';

            record_measurement(buffer, measurement);
        } else {
            receive_probe(socketFd, &recvTimeB, probeIndex);
            if (recvTimeB == 0) {
                ++skippedIterations;
                continue;
            }

            if (!send_probe(socketFd, &peerAddress, &sendTimeB, probeIndex)) {
                ++skippedIterations;
                continue;
            }

            sockaddr_in senderAddress{};
            socklen_t senderLength = sizeof(senderAddress);
            const ssize_t receivedBytes = recvfrom(socketFd,
                                                   messageBuffer,
                                                   sizeof(messageBuffer) - 1,
                                                   0,
                                                   reinterpret_cast<sockaddr*>(&senderAddress),
                                                   &senderLength);
            if (receivedBytes < 0) {
                ++skippedIterations;
                continue;
            }

            messageBuffer[receivedBytes] = '\0';
            if (std::sscanf(messageBuffer, "%lld %lld",
                            reinterpret_cast<long long*>(&sendTimeA),
                            reinterpret_cast<long long*>(&recvTimeA)) != 2) {
                ++skippedIterations;
                continue;
            }

            if (std::snprintf(messageBuffer, sizeof(messageBuffer), "%lld %lld",
                              static_cast<long long>(sendTimeB),
                              static_cast<long long>(recvTimeB)) < 0) {
                ++skippedIterations;
                continue;
            }

            if (sendto(socketFd,
                       messageBuffer,
                       std::strlen(messageBuffer),
                       0,
                       reinterpret_cast<sockaddr*>(&peerAddress),
                       sizeof(peerAddress)) < 0) {
                ++skippedIterations;
                continue;
            }

            const timestamp_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
            const int64_t     offset = (recvTimeA - sendTimeB) - (roundTripTime / 2);

            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            Measurement measurement{};
            measurement.timestampNs = timespec_to_ns(now);
            measurement.sendTimeA = sendTimeA;
            measurement.recvTimeA = recvTimeA;
            measurement.sendTimeB = sendTimeB;
            measurement.recvTimeB = recvTimeB;
            measurement.roundTripTime = roundTripTime;
            measurement.offset = offset;
            measurement.udpPort = udpPort;
            measurement.node[0] = 'B';
            measurement.node[1] = '\0';

            record_measurement(buffer, measurement);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    close(socketFd);
//    std::fprintf(stderr, "ChronoSync: [run_node] INFO - Node %s completed, skipped %d iterations\n", role, skippedIterations);
}

} // namespace firefly
} // namespace rpdtracer
