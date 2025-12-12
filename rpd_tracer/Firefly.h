#ifndef FIREFLY_H
#define FIREFLY_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <ctime>
// Standard C++ headers
#include <atomic>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>

// POSIX/System headers
#include <sys/socket.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <semaphore.h>
#include <time.h>
#include <errno.h>
#include <linux/net_tstamp.h>
#include <sys/types.h>
#include "Utility.h"


// Forward declarations for POSIX types
struct sockaddr_in;
// struct sem_t;

namespace firefly {

// -----------------------------------------------------------------------------
// Type Definitions
// -----------------------------------------------------------------------------
using timestamp_t = uint64_t;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int TCP_PORT_DEFAULT = 12345;
constexpr int UDP_PORT_DEFAULT = 12345;
constexpr std::size_t NETWORK_BUFFER_SIZE = 1024U;
constexpr int SOCKET_TIMEOUT_MSEC = 200;
constexpr int CONNECTION_RETRY_LIMIT = 30;
constexpr int CONNECTION_RETRY_DELAY_SEC = 1;
constexpr std::size_t MAX_MEASUREMENT_COUNT = 100000U;
constexpr std::size_t REGRESSION_WINDOW_SIZE = 1000U;
constexpr double CONSENSUS_ALPHA = 0.5;
constexpr double GAIN_PHASE = 1.0;
constexpr double GAIN_FREQ = 1.0;
constexpr int FIRE_FLY_SLEEP_MSEC = 100;
constexpr int SHM_PERMISSIONS = 0666;
constexpr int SEMAPHORE_SHARED_PROCESS = 1;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/**
 * @brief Represents a single timing measurement between two nodes.
 */
struct Measurement {
    timestamp_t timestampNs;       ///< Timestamp when measurement was recorded
    timestamp_t sendTimeA;         ///< Send time from node A
    timestamp_t recvTimeA;         ///< Receive time at node A
    timestamp_t sendTimeB;         ///< Send time from node B
    timestamp_t recvTimeB;         ///< Receive time at node B
    timestamp_t roundTripTime;     ///< Calculated round-trip time
    int64_t     offset;            ///< Clock offset between nodes
    int         udpPort;           ///< UDP port used for measurement
    char        node[2];           ///< Node identifier ('A' or 'B')
};

/**
 * @brief Service state for clock synchronization.
 */
struct SvcState {
    std::atomic<unsigned int> sequence{0};  ///< Sequence number for lock-free updates
    timespec referenceTime{};               ///< Reference timestamp
    int64_t  offset{0};                     ///< Current clock offset in nanoseconds
    double   drift{0.0};                    ///< Clock drift rate
};

/**
 * @brief Analysis results from measurement data.
 */
struct MeasurementAnalysis {
    std::vector<Measurement> samples;     ///< Filtered measurement samples
    int64_t averageOffset{0};             ///< Average offset in nanoseconds
    double  driftRate{0.0};               ///< Drift rate (ns/ns)
};

/**
 * @brief Manages shared memory region for inter-process communication.
 */
struct SharedMemoryRegion {
    /**
     * @brief Initialize shared memory region.
     * @param measurementCapacity Maximum number of measurements to store
     * @return true if initialization succeeded, false otherwise
     */
    bool initialize(std::size_t measurementCapacity);

    /**
     * @brief Get pointer to measurement array.
     * @return Pointer to measurements in shared memory
     */
    Measurement* measurements() const;

    /**
     * @brief Get pointer to measurement count.
     * @return Pointer to count of stored measurements
     */
    int* measurementCount() const;

    /**
     * @brief Get pointer to semaphore.
     * @return Pointer to semaphore for synchronization
     */
    sem_t* semaphore() const;

    /**
     * @brief Get shared memory ID.
     * @return Shared memory identifier
     */
    int id() const;

    /**
     * @brief Detach from shared memory without destroying it.
     */
    void detachWithoutDestroy() const;

    /**
     * @brief Destructor - cleans up shared memory resources.
     */
    ~SharedMemoryRegion();

private:
    int m_shmId{-1};                        ///< Shared memory ID
    void* m_address{nullptr};               ///< Shared memory address
    std::size_t m_measurementBytes{0};      ///< Size of measurement array
    std::size_t m_totalBytes{0};            ///< Total shared memory size
    pid_t m_ownerPid{0};                    ///< Process ID of owner
};

// -----------------------------------------------------------------------------
// Global State
// -----------------------------------------------------------------------------
extern SvcState g_svcState;

// -----------------------------------------------------------------------------
// Function Declarations
// -----------------------------------------------------------------------------

/**
 * @brief Enable software timestamps on a socket.
 * @param sockFd Socket file descriptor
 */
void enable_sw_timestamps(int sockFd);

/**
 * @brief Send a timing probe packet.
 * @param sockFd Socket file descriptor
 * @param peerAddress Destination address
 * @param sendTime Output parameter for send timestamp
 * @param probeId Probe identifier
 */
void send_probe(int sockFd, sockaddr_in* peerAddress, timestamp_t* sendTime, int probeId);

/**
 * @brief Receive a timing probe packet.
 * @param sockFd Socket file descriptor
 * @param receiveTime Output parameter for receive timestamp
 * @param expectedProbeId Expected probe identifier
 */
void receive_probe(int sockFd, timestamp_t* receiveTime, int expectedProbeId);

/**
 * @brief Perform TCP handshake between nodes.
 * @param role Node role ("A" or "B")
 * @param peerIp Peer IP address
 * @param tcpPort TCP port number
 * @return Socket file descriptor, or -1 on failure
 */
int tcp_handshake(const char* role, const char* peerIp, int tcpPort);

/**
 * @brief Run the synchronization node.
 * @param role Node role ("A" or "B")
 * @param ipA IP address of node A
 * @param ipB IP address of node B
 * @param udpPort UDP port for communication
 * @param sharedMeasurements Shared memory for measurements
 * @param sharedCount Shared counter for measurements
 * @param semaphoreHandle Semaphore for synchronization
 */
void run_node(const char* role,
              const char* ipA,
              const char* ipB,
              int udpPort,
              Measurement* sharedMeasurements,
              int* sharedCount,
              sem_t* semaphoreHandle);

/**
 * @brief Run the Firefly synchronization algorithm.
 * @param role Node role ("A" or "B")
 * @param sharedMeasurements Shared memory for measurements
 * @param sharedCount Shared counter for measurements
 * @param semaphoreHandle Semaphore for synchronization
 */
void firefly_run(const char* role,
                 Measurement* sharedMeasurements,
                 int* sharedCount,
                 sem_t* semaphoreHandle);

/**
 * @brief Get current hardware timestamp.
 * @return Current time as timespec
 */
timespec hw_now();

/**
 * @brief Update service state with new offset and drift.
 * @param state Pointer to service state
 * @param offsetNs Clock offset in nanoseconds
 * @param drift Clock drift rate
 */
void svc_update_ns(SvcState* state, int64_t offsetNs, double drift);

/**
 * @brief Extract and analyze measurements for a specific role.
 * @param role Node role filter ("A" or "B")
 * @param windowSize Maximum number of samples to examine
 * @param sharedMeasurements Shared memory measurement buffer
 * @param sharedCount Total stored measurement count
 * @return MeasurementAnalysis containing samples, average offset, and drift rate
 */
MeasurementAnalysis read_latest_measurements(const char* role,
                                             std::size_t windowSize,
                                             const Measurement* sharedMeasurements,
                                             int sharedCount);
} // namespace firefly

#endif // FIREFLY_H
