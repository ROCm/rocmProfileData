#include "Firefly.h"

namespace firefly {

// -----------------------------------------------------------------------------
// Global State Definition
// -----------------------------------------------------------------------------
SvcState g_svcState;

// -----------------------------------------------------------------------------
// SharedMemoryRegion Implementation
// -----------------------------------------------------------------------------
bool SharedMemoryRegion::initialize(std::size_t measurementCapacity) {
    m_ownerPid = getpid();
    m_measurementBytes = measurementCapacity * sizeof(Measurement);
    m_totalBytes = m_measurementBytes + sizeof(int) + sizeof(sem_t);

    m_shmId = shmget(IPC_PRIVATE, m_totalBytes, IPC_CREAT | SHM_PERMISSIONS);
    if (m_shmId < 0) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::initialize] CRITICAL - shmget failed (errno: %s)\n", std::strerror(errno));
        return false;
    }

    m_address = shmat(m_shmId, nullptr, 0);
    if (m_address == reinterpret_cast<void*>(-1)) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::initialize] CRITICAL - shmat failed (errno: %s)\n", std::strerror(errno));
        m_address = nullptr;
        shmctl(m_shmId, IPC_RMID, nullptr);
        m_shmId = -1;
        return false;
    }

    std::memset(m_address, 0, m_totalBytes);
    *measurementCount() = 0;

    if (sem_init(semaphore(), SEMAPHORE_SHARED_PROCESS, 1) == -1) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::initialize] CRITICAL - sem_init failed (errno: %s)\n", std::strerror(errno));
        shmdt(m_address);
        shmctl(m_shmId, IPC_RMID, nullptr);
        m_address = nullptr;
        m_shmId = -1;
        return false;
    }

    return true;
}

Measurement* SharedMemoryRegion::measurements() const {
    return static_cast<Measurement*>(m_address);
}

int* SharedMemoryRegion::measurementCount() const {
    return reinterpret_cast<int*>(static_cast<char*>(m_address) + m_measurementBytes);
}

sem_t* SharedMemoryRegion::semaphore() const {
    return reinterpret_cast<sem_t*>(reinterpret_cast<char*>(m_address) + m_measurementBytes + sizeof(int));
}

int SharedMemoryRegion::id() const {
    return m_shmId;
}

void SharedMemoryRegion::detachWithoutDestroy() const {
    if (m_address != nullptr) {
        shmdt(m_address);
    }
}

SharedMemoryRegion::~SharedMemoryRegion() {
    if (m_address == nullptr || m_ownerPid != getpid()) {
        if (m_address != nullptr) {
            shmdt(m_address);
        }
        return;
    }

    if (sem_destroy(semaphore()) == -1) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::~SharedMemoryRegion] WARNING - sem_destroy failed (errno: %s)\n", std::strerror(errno));
    }

    if (shmdt(m_address) == -1) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::~SharedMemoryRegion] WARNING - shmdt failed (errno: %s)\n", std::strerror(errno));
    }

    if (shmctl(m_shmId, IPC_RMID, nullptr) == -1) {
        std::fprintf(stderr, "ChronoSync: [SharedMemoryRegion::~SharedMemoryRegion] WARNING - shmctl failed (errno: %s)\n", std::strerror(errno));
    }
}

// -----------------------------------------------------------------------------
// Measurement Helpers
// -----------------------------------------------------------------------------
MeasurementAnalysis read_latest_measurements(const char* role,
                                             std::size_t windowSize,
                                             const Measurement* sharedMeasurements,
                                             int sharedCount) {
    MeasurementAnalysis analysis;
    if (role == nullptr || sharedMeasurements == nullptr || sharedCount <= 0) {
        return analysis;
    }

    analysis.samples.reserve(static_cast<std::size_t>(sharedCount));
    for (int index = 0; index < sharedCount; ++index) {
        if (std::strncmp(sharedMeasurements[index].node, role, sizeof(sharedMeasurements[index].node)) == 0) {
            analysis.samples.push_back(sharedMeasurements[index]);
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

    double sumX = 0.0;
    int64_t sumY = 0.0;
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
    
    analysis.averageOffset = sumY / sampleCount;
    analysis.driftRate = slope;

    fprintf(stderr,
                 "ChronoSync: [read_latest_measurements] INFO - Role %s, Avg Offset: %lld ns, Drift: %.9e ns/ns\n",
                 role,
                 static_cast<int64_t>(analysis.averageOffset),
                 static_cast<double>(analysis.driftRate));

    return analysis;
}

// -----------------------------------------------------------------------------
// Networking Utilities
// -----------------------------------------------------------------------------
void enable_sw_timestamps(int sockFd) {
    const int timestampOptions = SOF_TIMESTAMPING_TX_SOFTWARE |
                                 SOF_TIMESTAMPING_RX_SOFTWARE |
                                 SOF_TIMESTAMPING_SOFTWARE |
                                 SOF_TIMESTAMPING_OPT_TSONLY;

    if (setsockopt(sockFd, SOL_SOCKET, SO_TIMESTAMPING, &timestampOptions, sizeof(timestampOptions)) < 0) {
        std::fprintf(stderr, "ChronoSync: [enable_sw_timestamps] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
}

void send_probe(int sockFd, sockaddr_in* peerAddress, timestamp_t* sendTime, int probeId) {
    if (peerAddress == nullptr || sendTime == nullptr) {
        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - Invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }

    char messageBuffer[NETWORK_BUFFER_SIZE] = {};
    if (std::snprintf(messageBuffer, sizeof(messageBuffer), "Probe %d", probeId) < 0) {
        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - snprintf failed\n");
        std::exit(EXIT_FAILURE);
    }

    if (sendto(sockFd,
               messageBuffer,
               std::strlen(messageBuffer),
               0,
               reinterpret_cast<sockaddr*>(peerAddress),
               sizeof(*peerAddress)) < 0) {
        std::fprintf(stderr, "ChronoSync: [send_probe] CRITICAL - sendto failed (errno: %s)\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC, &timeSpec) == 0) {
        *sendTime = timespec_to_ns(timeSpec);
    } else {
        std::fprintf(stderr, "ChronoSync: [send_probe] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
        *sendTime = 0;
    }
}

void receive_probe(int sockFd, timestamp_t* receiveTime, int expectedProbeId) {
    if (receiveTime == nullptr) {
        std::fprintf(stderr, "ChronoSync: [receive_probe] CRITICAL - Invalid receiveTime pointer\n");
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
        std::fprintf(stderr, "ChronoSync: [receive_probe] WARNING - recvmsg failed (errno: %s)\n", std::strerror(errno));
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
        std::fprintf(stderr, "ChronoSync: [receive_probe] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
        *receiveTime = 0;
    }
}

int tcp_handshake(const char* role, const char* peerIp, int tcpPort) {
    if (role == nullptr || peerIp == nullptr) {
        std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - Invalid role or peerIp\n");
        return -1;
    }

    const bool isNodeA = (std::strcmp(role, "A") == 0);
    int tcpSocketFd = -1;
    sockaddr_in tcpAddress{};
    tcpAddress.sin_family = AF_INET;
    tcpAddress.sin_port = htons(tcpPort);

    if (isNodeA) {
        tcpSocketFd = socket(AF_INET, SOCK_STREAM, 0);
        if (tcpSocketFd < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }

        if (inet_aton(peerIp, &tcpAddress.sin_addr) == 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - Invalid peer IP\n");
            close(tcpSocketFd);
            std::exit(EXIT_FAILURE);
        }

        bool connected = false;
        for (int attempt = 0; attempt < CONNECTION_RETRY_LIMIT; ++attempt) {
            if (connect(tcpSocketFd, reinterpret_cast<sockaddr*>(&tcpAddress), sizeof(tcpAddress)) == 0) {
                connected = true;
                break;
            }
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] WARNING - connect attempt %d failed (errno: %s)\n",
                         attempt + 1,
                         std::strerror(errno));
            sleep(CONNECTION_RETRY_DELAY_SEC);
        }

        if (!connected) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - Unable to connect after retries\n");
            close(tcpSocketFd);
            std::exit(EXIT_FAILURE);
        }

        constexpr char HANDSHAKE_READY[] = "READY";
        if (send(tcpSocketFd, HANDSHAKE_READY, sizeof(HANDSHAKE_READY), 0) < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - send failed (errno: %s)\n", std::strerror(errno));
            close(tcpSocketFd);
            std::exit(EXIT_FAILURE);
        }

        char ackBuffer[NETWORK_BUFFER_SIZE] = {};
        const ssize_t received = recv(tcpSocketFd, ackBuffer, sizeof(ackBuffer) - 1, 0);
        if (received <= 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] WARNING - recv failed (errno: %s)\n", std::strerror(errno));
        } else {
            ackBuffer[received] = '\0';
        }
    } else {
        int listenSocketFd = socket(AF_INET, SOCK_STREAM, 0);
        if (listenSocketFd < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }

        int reuseAddress = 1;
        if (setsockopt(listenSocketFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddress, sizeof(reuseAddress)) < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            std::exit(EXIT_FAILURE);
        }

        tcpAddress.sin_addr.s_addr = INADDR_ANY;
        if (bind(listenSocketFd, reinterpret_cast<sockaddr*>(&tcpAddress), sizeof(tcpAddress)) < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - bind failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            std::exit(EXIT_FAILURE);
        }

        if (listen(listenSocketFd, 1) < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - listen failed (errno: %s)\n", std::strerror(errno));
            close(listenSocketFd);
            std::exit(EXIT_FAILURE);
        }

        sockaddr_in clientAddress{};
        socklen_t clientLength = sizeof(clientAddress);
        tcpSocketFd = accept(listenSocketFd, reinterpret_cast<sockaddr*>(&clientAddress), &clientLength);
        close(listenSocketFd);

        if (tcpSocketFd < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - accept failed (errno: %s)\n", std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }

        char handshakeBuffer[NETWORK_BUFFER_SIZE] = {};
        const ssize_t received = recv(tcpSocketFd, handshakeBuffer, sizeof(handshakeBuffer) - 1, 0);
        if (received > 0) {
            handshakeBuffer[received] = '\0';
        }

        constexpr char HANDSHAKE_ACK[] = "ACK";
        if (send(tcpSocketFd, HANDSHAKE_ACK, sizeof(HANDSHAKE_ACK), 0) < 0) {
            std::fprintf(stderr, "ChronoSync: [tcp_handshake] CRITICAL - send failed (errno: %s)\n", std::strerror(errno));
            close(tcpSocketFd);
            std::exit(EXIT_FAILURE);
        }
    }

    return tcpSocketFd;
}

// -----------------------------------------------------------------------------
// Clock Helpers
// -----------------------------------------------------------------------------

timespec hw_now() {
    timespec timeSpec{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timeSpec) != 0) {
        std::fprintf(stderr, "ChronoSync: [hw_now] WARNING - clock_gettime failed (errno: %s)\n", std::strerror(errno));
    }
    return timeSpec;
}

void svc_update_ns(SvcState* state, int64_t offsetNs, double drift) {
    if (state == nullptr) {
        std::fprintf(stderr, "ChronoSync: [svc_update_ns] CRITICAL - Null svc state\n");
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
void firefly_run(const char* role,
                 Measurement* sharedMeasurements,
                 int* sharedCount,
                 sem_t* semaphoreHandle) {
    if (role == nullptr || sharedMeasurements == nullptr || sharedCount == nullptr || semaphoreHandle == nullptr) {
        std::fprintf(stderr, "ChronoSync: [firefly_run] CRITICAL - Invalid arguments\n");
        return;
    }

    int localCount = 0;
    if (sem_wait(semaphoreHandle) == -1) {
        std::fprintf(stderr, "ChronoSync: [firefly_run] WARNING - sem_wait failed (errno: %s)\n", std::strerror(errno));
        return;
    }
    localCount = *sharedCount;
    if (sem_post(semaphoreHandle) == -1) {
        std::fprintf(stderr, "ChronoSync: [firefly_run] WARNING - sem_post failed (errno: %s)\n", std::strerror(errno));
    }

    std::fprintf(stderr, "ChronoSync: [firefly_run] INFO - Role %s, shared count: %d\n", role, localCount);

    const MeasurementAnalysis analysis = read_latest_measurements(role,
                                                                  REGRESSION_WINDOW_SIZE,
                                                                  sharedMeasurements,
                                                                  localCount);

    std::fprintf(stderr,
                 "ChronoSync: [firefly_run] INFO - Role %s, Avg Offset: %lld ns, Drift: %.9e ns/ns\n",
                 role,
                 static_cast<int64_t>(analysis.averageOffset),
                 analysis.driftRate);

    if (!analysis.samples.empty()) {
        FILE* fileHandle = std::fopen("final_sync_params.txt", "a");
        if (fileHandle != nullptr) {
            std::fprintf(fileHandle,
                         "Role: %s, Average Offset: %lld ns, Drift Rate: %.9e ns/ns\n",
                         role,
                         static_cast<int64_t>(analysis.averageOffset),
                         analysis.driftRate);
            std::fclose(fileHandle);
        } else {
            std::fprintf(stderr,
                         "ChronoSync: [firefly_run] WARNING - fopen failed (errno: %s)\n",
                         std::strerror(errno));
        }
    }

    svc_update_ns(&g_svcState, static_cast<int64_t>(analysis.averageOffset * CONSENSUS_ALPHA), analysis.driftRate * CONSENSUS_ALPHA);
    std::fprintf(stderr,
                 "ChronoSync: [firefly_run] INFO - Updated svc offset=%lld drift=%.9e\n",
                 g_svcState.offset,
                 g_svcState.drift);

    sync_clock(g_svcState.offset, g_svcState.drift);
}

// -----------------------------------------------------------------------------
// UDP Node
// -----------------------------------------------------------------------------
void run_node(const char* role,
              const char* ipA,
              const char* ipB,
              int udpPort,
              Measurement* sharedMeasurements,
              int* sharedCount,
              sem_t* semaphoreHandle) {
    if (role == nullptr || ipA == nullptr || ipB == nullptr || sharedMeasurements == nullptr ||
        sharedCount == nullptr || semaphoreHandle == nullptr) {
        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - Invalid arguments\n");
        std::exit(EXIT_FAILURE);
    }

    const bool isNodeA = (std::strcmp(role, "A") == 0);
    int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd < 0) {
        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - socket failed (errno: %s)\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    timeval timeoutValue{};
    timeoutValue.tv_sec = SOCKET_TIMEOUT_MSEC / 1000;
    timeoutValue.tv_usec = (SOCKET_TIMEOUT_MSEC % 1000) * 1000;
    if (setsockopt(socketFd, SOL_SOCKET, SO_RCVTIMEO, &timeoutValue, sizeof(timeoutValue)) < 0) {
        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - setsockopt failed (errno: %s)\n", std::strerror(errno));
        close(socketFd);
        std::exit(EXIT_FAILURE);
    }

    enable_sw_timestamps(socketFd);

    sockaddr_in localAddress{};
    sockaddr_in peerAddress{};
    localAddress.sin_family = AF_INET;
    peerAddress.sin_family = AF_INET;
    localAddress.sin_port = htons(udpPort);
    peerAddress.sin_port = htons(udpPort);

    const char* localIp = isNodeA ? ipA : ipB;
    const char* peerIp = isNodeA ? ipB : ipA;

    if (inet_aton(localIp, &localAddress.sin_addr) == 0 || inet_aton(peerIp, &peerAddress.sin_addr) == 0) {
        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - Invalid IP address\n");
        close(socketFd);
        std::exit(EXIT_FAILURE);
    }

    if (bind(socketFd, reinterpret_cast<sockaddr*>(&localAddress), sizeof(localAddress)) < 0) {
        std::fprintf(stderr, "ChronoSync: [run_node] CRITICAL - bind failed (errno: %s)\n", std::strerror(errno));
        close(socketFd);
        std::exit(EXIT_FAILURE);
    }

    int tcpSocketFd = tcp_handshake(role, peerIp, udpPort);
    close(tcpSocketFd);

    char messageBuffer[NETWORK_BUFFER_SIZE];
    int skippedIterations = 0;
    int probeIndex = 0;

    while (true) {
        ++probeIndex;
        timestamp_t sendTimeA = 0;
        timestamp_t recvTimeA = 0;
        timestamp_t sendTimeB = 0;
        timestamp_t recvTimeB = 0;

        if (isNodeA) {
            send_probe(socketFd, &peerAddress, &sendTimeA, probeIndex);
            receive_probe(socketFd, &recvTimeA, probeIndex);
            if (recvTimeA == 0) {
                ++skippedIterations;
                continue;
            }

            if (std::snprintf(messageBuffer, sizeof(messageBuffer), "%lld %lld",
                              static_cast<long long>(sendTimeA),
                              static_cast<long long>(recvTimeA)) < 0) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - snprintf failed\n");
                continue;
            }

            if (sendto(socketFd, messageBuffer, std::strlen(messageBuffer), 0,
                       reinterpret_cast<sockaddr*>(&peerAddress), sizeof(peerAddress)) < 0) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sendto failed (errno: %s)\n", std::strerror(errno));
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
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - recvfrom failed (errno: %s)\n", std::strerror(errno));
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
            Measurement measurement{
                .timestampNs = timespec_to_ns(now),
                .sendTimeA = sendTimeA,
                .recvTimeA = recvTimeA,
                .sendTimeB = sendTimeB,
                .recvTimeB = recvTimeB,
                .roundTripTime = roundTripTime,
                .offset = offset,
                .udpPort = udpPort,
                .node = {'A', '\0'}
            };

            if (sem_wait(semaphoreHandle) == -1) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sem_wait failed (errno: %s)\n", std::strerror(errno));
                continue;
            }

            if (*sharedCount < static_cast<int>(MAX_MEASUREMENT_COUNT)) {
                sharedMeasurements[*sharedCount] = measurement;
                ++(*sharedCount);
            }

            if (sem_post(semaphoreHandle) == -1) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sem_post failed (errno: %s)\n", std::strerror(errno));
            }
        } else {
            receive_probe(socketFd, &recvTimeB, probeIndex);
            if (recvTimeB == 0) {
                ++skippedIterations;
                continue;
            }

            send_probe(socketFd, &peerAddress, &sendTimeB, probeIndex);

            sockaddr_in senderAddress{};
            socklen_t senderLength = sizeof(senderAddress);
            const ssize_t receivedBytes = recvfrom(socketFd,
                                                   messageBuffer,
                                                   sizeof(messageBuffer) - 1,
                                                   0,
                                                   reinterpret_cast<sockaddr*>(&senderAddress),
                                                   &senderLength);
            if (receivedBytes < 0) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - recvfrom failed (errno: %s)\n", std::strerror(errno));
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
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sendto failed (errno: %s)\n", std::strerror(errno));
                ++skippedIterations;
                continue;
            }

            const timestamp_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
            const int64_t     offset = (recvTimeA - sendTimeB) - (roundTripTime / 2);

            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            Measurement measurement{
                .timestampNs = timespec_to_ns(now),
                .sendTimeA = sendTimeA,
                .recvTimeA = recvTimeA,
                .sendTimeB = sendTimeB,
                .recvTimeB = recvTimeB,
                .roundTripTime = roundTripTime,
                .offset = offset,
                .udpPort = udpPort,
                .node = {'B', '\0'}
            };

            if (sem_wait(semaphoreHandle) == -1) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sem_wait failed (errno: %s)\n", std::strerror(errno));
                continue;
            }

            if (*sharedCount < static_cast<int>(MAX_MEASUREMENT_COUNT)) {
                sharedMeasurements[*sharedCount] = measurement;
                ++(*sharedCount);
            }

            if (sem_post(semaphoreHandle) == -1) {
                std::fprintf(stderr, "ChronoSync: [run_node] WARNING - sem_post failed (errno: %s)\n", std::strerror(errno));
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    close(socketFd);
    std::fprintf(stderr, "ChronoSync: [run_node] INFO - Node %s completed, skipped %d iterations\n", role, skippedIterations);
}

} // namespace firefly

