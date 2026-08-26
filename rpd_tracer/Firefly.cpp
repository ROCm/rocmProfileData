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
                                             int count,
                                             std::size_t capacity) {
    MeasurementAnalysis analysis;
    if (role == nullptr || measurements == nullptr || count <= 0) {
        return analysis;
    }

    int available = (count < static_cast<int>(capacity)) ? count : static_cast<int>(capacity);
    int start = count - available;
    analysis.samples.reserve(static_cast<std::size_t>(available));
    for (int i = 0; i < available; ++i) {
        int index = (start + i) % static_cast<int>(capacity);
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
static bool enable_sw_timestamps(int sockFd) {
    const int timestampOptions = SOF_TIMESTAMPING_TX_SOFTWARE |
                                 SOF_TIMESTAMPING_RX_SOFTWARE |
                                 SOF_TIMESTAMPING_SOFTWARE |
                                 SOF_TIMESTAMPING_OPT_TSONLY;

    if (setsockopt(sockFd, SOL_SOCKET, SO_TIMESTAMPING, &timestampOptions, sizeof(timestampOptions)) < 0)
        return false;
    return true;
}

static bool send_probe(int sockFd, sockaddr_in* peerAddress, timestamp_t* sendTime, int probeId) {
    if (peerAddress == nullptr || sendTime == nullptr)
        return false;

    char messageBuffer[NETWORK_BUFFER_SIZE] = {};
    if (std::snprintf(messageBuffer, sizeof(messageBuffer), "Probe %d", probeId) < 0)
        return false;

    if (sendto(sockFd, messageBuffer, std::strlen(messageBuffer), 0,
               reinterpret_cast<sockaddr*>(peerAddress), sizeof(*peerAddress)) < 0)
        return false;

    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC, &timeSpec) == 0) {
        *sendTime = timespec_to_ns(timeSpec);
    } else {
        *sendTime = 0;
    }
    return true;
}

static void receive_probe(int sockFd, timestamp_t* receiveTime, int expectedProbeId) {
    if (receiveTime == nullptr)
        return;

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
        *receiveTime = 0;
    }
}

static void record_measurement(MeasurementBuffer& buffer, const Measurement& measurement) {
    std::lock_guard<std::mutex> lock(buffer.mutex);
    int idx = buffer.count % static_cast<int>(buffer.measurements.size());
    buffer.measurements[idx] = measurement;
    ++buffer.count;
}

// -----------------------------------------------------------------------------
// Clock Helpers
// -----------------------------------------------------------------------------

timespec hw_now() {
    timespec timeSpec{};
    clock_gettime(CLOCK_MONOTONIC, &timeSpec);
    return timeSpec;
}

void svc_update_ns(SvcState* state, int64_t offsetNs, double drift) {
    if (state == nullptr)
        return;

    const unsigned sequence = state->sequence.fetch_add(1U, std::memory_order_acq_rel);
    state->referenceTime = hw_now();
    state->offset = offsetNs;
    state->drift = drift;
    state->sequence.store(sequence + 2U, std::memory_order_release);
}

// -----------------------------------------------------------------------------
// PI Controller State
// -----------------------------------------------------------------------------
struct PidState {
    double frequency{0.0};
    bool initialized{false};
};

static PidState s_pid;

void firefly_reset() {
    s_pid = PidState{};
}

// -----------------------------------------------------------------------------
// Firefly — PI-based clock offset tracking
// -----------------------------------------------------------------------------
void firefly_run(const char* role, MeasurementBuffer& buffer) {
    if (role == nullptr)
        return;

    int localCount = 0;
    int64_t latest_offset = 0;
    {
        std::lock_guard<std::mutex> lock(buffer.mutex);
        localCount = buffer.count;
        if (localCount <= 0)
            return;
        int idx = (localCount - 1) % static_cast<int>(buffer.measurements.size());
        latest_offset = buffer.measurements[idx].offset;
    }

    // IQR outlier check: reject the latest sample if it's an outlier
    // relative to the recent window
    if (localCount > static_cast<int>(MEDIAN_WINDOW_SIZE)) {
        const MeasurementAnalysis analysis = read_latest_measurements(role,
                                                                      MEDIAN_WINDOW_SIZE,
                                                                      buffer.measurements.data(),
                                                                      localCount,
                                                                      buffer.measurements.size());
        if (!analysis.samples.empty()) {
            std::vector<int64_t> offsets;
            offsets.reserve(analysis.samples.size());
            for (const auto& s : analysis.samples)
                offsets.push_back(s.offset);
            std::sort(offsets.begin(), offsets.end());
            size_t n = offsets.size();
            int64_t q1 = offsets[n / 4];
            int64_t q3 = offsets[3 * n / 4];
            int64_t iqr = q3 - q1;
            if (latest_offset < q1 - 3 * iqr || latest_offset > q3 + 3 * iqr)
                return;
        }
    }

    int64_t measured_offset = latest_offset;

    if (!s_pid.initialized) {
        svc_update_ns(firefly::g_pSvcState, measured_offset, 0.0);
        s_pid.initialized = true;
        return;
    }

    double current_offset = static_cast<double>(g_pSvcState->offset);
    double error = static_cast<double>(measured_offset) - current_offset;
    double dt_ns = FIRE_FLY_SLEEP_MSEC * 1e6;

    // I term: learn drift rate from persistent phase error
    s_pid.frequency += KI * error / dt_ns;

    // Anti-windup: clamp at 500 ppm
    if (s_pid.frequency > 500e-6)
        s_pid.frequency = 500e-6;
    if (s_pid.frequency < -500e-6)
        s_pid.frequency = -500e-6;

    // P term: phase correction + frequency correction
    double new_offset = current_offset + KP * error + s_pid.frequency * dt_ns;

    svc_update_ns(firefly::g_pSvcState, static_cast<int64_t>(new_offset), s_pid.frequency);
}

// -----------------------------------------------------------------------------
// Hub-and-Spoke: Probe Exchange
// -----------------------------------------------------------------------------
void run_probes(const char* role, const char* peerIp, int udpPort,
                MeasurementBuffer& buffer, std::atomic<bool>& done) {
    if (role == nullptr || peerIp == nullptr)
        return;

    const bool isNodeA = (std::strcmp(role, "A") == 0);
    int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd < 0)
        return;

    timeval timeoutValue{};
    timeoutValue.tv_sec = SOCKET_TIMEOUT_MSEC / 1000;
    timeoutValue.tv_usec = (SOCKET_TIMEOUT_MSEC % 1000) * 1000;
    if (setsockopt(socketFd, SOL_SOCKET, SO_RCVTIMEO, &timeoutValue, sizeof(timeoutValue)) < 0) {
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
    localAddress.sin_addr.s_addr = INADDR_ANY;
    localAddress.sin_port = htons(udpPort);
    peerAddress.sin_family = AF_INET;
    peerAddress.sin_port = htons(udpPort);

    if (inet_aton(peerIp, &peerAddress.sin_addr) == 0) {
        close(socketFd);
        return;
    }

    if (bind(socketFd, reinterpret_cast<sockaddr*>(&localAddress), sizeof(localAddress)) < 0) {
        close(socketFd);
        return;
    }

    char messageBuffer[NETWORK_BUFFER_SIZE];
    int probeIndex = 0;

    while (!done.load(std::memory_order_relaxed)) {
        ++probeIndex;
        timestamp_t sendTimeA = 0;
        timestamp_t recvTimeA = 0;
        timestamp_t sendTimeB = 0;
        timestamp_t recvTimeB = 0;

        if (isNodeA) {
            if (!send_probe(socketFd, &peerAddress, &sendTimeA, probeIndex))
                continue;
            receive_probe(socketFd, &recvTimeA, probeIndex);
            if (recvTimeA == 0)
                continue;

            std::snprintf(messageBuffer, sizeof(messageBuffer), "%lld %lld",
                          static_cast<long long>(sendTimeA),
                          static_cast<long long>(recvTimeA));
            if (sendto(socketFd, messageBuffer, std::strlen(messageBuffer), 0,
                       reinterpret_cast<sockaddr*>(&peerAddress), sizeof(peerAddress)) < 0)
                continue;

            sockaddr_in senderAddress{};
            socklen_t senderLength = sizeof(senderAddress);
            const ssize_t receivedBytes = recvfrom(socketFd, messageBuffer,
                                                   sizeof(messageBuffer) - 1, 0,
                                                   reinterpret_cast<sockaddr*>(&senderAddress),
                                                   &senderLength);
            if (receivedBytes < 0)
                continue;

            messageBuffer[receivedBytes] = '\0';
            if (std::sscanf(messageBuffer, "%lld %lld",
                            reinterpret_cast<long long*>(&sendTimeB),
                            reinterpret_cast<long long*>(&recvTimeB)) != 2)
                continue;

            const timestamp_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
            const int64_t offset = (recvTimeB - sendTimeA) - (roundTripTime / 2);

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
            if (recvTimeB == 0)
                continue;

            if (!send_probe(socketFd, &peerAddress, &sendTimeB, probeIndex))
                continue;

            sockaddr_in senderAddress{};
            socklen_t senderLength = sizeof(senderAddress);
            const ssize_t receivedBytes = recvfrom(socketFd, messageBuffer,
                                                   sizeof(messageBuffer) - 1, 0,
                                                   reinterpret_cast<sockaddr*>(&senderAddress),
                                                   &senderLength);
            if (receivedBytes < 0)
                continue;

            messageBuffer[receivedBytes] = '\0';
            if (std::sscanf(messageBuffer, "%lld %lld",
                            reinterpret_cast<long long*>(&sendTimeA),
                            reinterpret_cast<long long*>(&recvTimeA)) != 2)
                continue;

            std::snprintf(messageBuffer, sizeof(messageBuffer), "%lld %lld",
                          static_cast<long long>(sendTimeB),
                          static_cast<long long>(recvTimeB));
            if (sendto(socketFd, messageBuffer, std::strlen(messageBuffer), 0,
                       reinterpret_cast<sockaddr*>(&peerAddress), sizeof(peerAddress)) < 0)
                continue;

            const timestamp_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
            const int64_t offset = (recvTimeA - sendTimeB) - (roundTripTime / 2);

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
}

// -----------------------------------------------------------------------------
// Hub-and-Spoke: Server (rank 0)
// -----------------------------------------------------------------------------
void run_server(int port, MeasurementBuffer& buffer, std::atomic<bool>& done) {
    int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0)
        return;

    int reuseAddr = 1;
    setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddr, sizeof(reuseAddr));

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);

    if (bind(listenFd, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        std::fprintf(stderr, "ChronoSync: server bind failed on port %d (errno: %s)\n",
                     port, std::strerror(errno));
        close(listenFd);
        return;
    }

    if (listen(listenFd, 16) < 0) {
        close(listenFd);
        return;
    }

    // Accept timeout so we can check the done flag
    timeval acceptTimeout{};
    acceptTimeout.tv_sec = 1;
    setsockopt(listenFd, SOL_SOCKET, SO_RCVTIMEO, &acceptTimeout, sizeof(acceptTimeout));

    std::fprintf(stderr, "ChronoSync: server listening on port %d\n", port);

    std::vector<std::thread> probeThreads;
    int clientCount = 0;

    while (!done.load(std::memory_order_relaxed)) {
        sockaddr_in clientAddr{};
        socklen_t clientLen = sizeof(clientAddr);
        int clientFd = accept(listenFd, reinterpret_cast<sockaddr*>(&clientAddr), &clientLen);
        if (clientFd < 0)
            continue;

        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, sizeof(clientIp));

        int udpPort = port + 1 + clientCount;

        // Send UDP port assignment to client
        char msg[32];
        std::snprintf(msg, sizeof(msg), "%d", udpPort);
        send(clientFd, msg, std::strlen(msg), 0);

        // Wait for ACK
        char ack[32] = {};
        recv(clientFd, ack, sizeof(ack) - 1, 0);
        close(clientFd);

        std::string peerIp(clientIp);
        probeThreads.emplace_back([peerIp, udpPort, &buffer, &done]() {
            run_probes("A", peerIp.c_str(), udpPort, buffer, done);
        });

        ++clientCount;
        std::fprintf(stderr, "ChronoSync: client %s connected (udp port %d)\n", clientIp, udpPort);
    }

    for (auto& t : probeThreads)
        t.join();
    close(listenFd);
}

// -----------------------------------------------------------------------------
// Hub-and-Spoke: Client (non-rank-0)
// -----------------------------------------------------------------------------
void run_client(const char* serverIp, int port,
                MeasurementBuffer& buffer, std::atomic<bool>& done) {
    if (serverIp == nullptr)
        return;

    int tcpFd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcpFd < 0)
        return;

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    if (inet_aton(serverIp, &serverAddr.sin_addr) == 0) {
        close(tcpFd);
        return;
    }

    bool connected = false;
    for (int attempt = 0; attempt < CONNECTION_RETRY_LIMIT && !done.load(std::memory_order_relaxed); ++attempt) {
        if (connect(tcpFd, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == 0) {
            connected = true;
            break;
        }
        sleep(CONNECTION_RETRY_DELAY_SEC);
    }

    if (!connected) {
        std::fprintf(stderr, "ChronoSync: failed to connect to %s:%d after %d attempts\n",
                     serverIp, port, CONNECTION_RETRY_LIMIT);
        close(tcpFd);
        return;
    }

    // Receive UDP port assignment from server
    char msg[32] = {};
    ssize_t received = recv(tcpFd, msg, sizeof(msg) - 1, 0);
    if (received <= 0) {
        close(tcpFd);
        return;
    }
    msg[received] = '\0';
    int udpPort = std::atoi(msg);

    // Send ACK
    send(tcpFd, "OK", 2, 0);
    close(tcpFd);

    std::fprintf(stderr, "ChronoSync: connected to %s (udp port %d)\n", serverIp, udpPort);

    run_probes("B", serverIp, udpPort, buffer, done);
}

} // namespace firefly
} // namespace rpdtracer
