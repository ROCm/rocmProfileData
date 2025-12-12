/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ChronoSyncDataSource.h"

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/net_tstamp.h>
#include <netinet/in.h>
#include <semaphore.h>
#include <sys/file.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include "rocm_smi/rocm_smi.h"

#include "Logger.h"
#include "Utility.h"
#include "Firefly.h"

constexpr char LOCK_FILE_PATH[] = "/tmp/chronosync_singleton.lock";
static int g_lockFd = -1;
// -----------------------------------------------------------------------------
// ChronoSyncDataSourcePrivate
// -----------------------------------------------------------------------------
class ChronoSyncDataSourcePrivate {
public:
    explicit ChronoSyncDataSourcePrivate(ChronoSyncDataSource* owner)
        : m_owner(owner) {}

    void work() {
        if (m_owner == nullptr) {
            std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSourcePrivate::work] CRITICAL - Owner not set\n");
            return;
        }

        std::unique_lock<std::mutex> lock(m_owner->m_mutex);
        while (!m_done.load(std::memory_order_relaxed)) {
            if (!m_owner->m_workExecuted) {
                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Executing work once\n");
                m_workerRunning = true;
                lock.unlock();
                m_owner->work();
                lock.lock();
                m_owner->m_workExecuted = true;
                m_workerRunning = false;
                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Work completed\n");
            }

            m_workerRunning = false;
            if (!m_done.load(std::memory_order_relaxed)) {
                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Waiting for signal\n");
                m_owner->m_wait.wait(lock);
            }
            m_workerRunning = true;
        }

        std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Exiting\n");
    }

    ChronoSyncDataSource* m_owner{nullptr};
    std::thread* m_worker{nullptr};
    std::atomic<bool> m_done{false};
    bool m_workerRunning{false};
    std::string m_hostIp;
    int m_rank{-1};
    std::vector<std::pair<std::string, int>> m_neighbors;
};

// -----------------------------------------------------------------------------
// ChronoSyncDataSource Implementation
// -----------------------------------------------------------------------------
extern "C" {
DataSource* ChronoSyncDataSourceFactory() {
    return new ChronoSyncDataSource();
}
}

ChronoSyncDataSource::ChronoSyncDataSource()
    : m_private(nullptr),
      m_workExecuted(false),
      m_messageCount(0) {}

ChronoSyncDataSource::~ChronoSyncDataSource() {
    end();
    delete m_private;
    m_private = nullptr;
}

bool ChronoSyncDataSource::tryAcquireGlobalLock() {
    g_lockFd = open(LOCK_FILE_PATH, O_CREAT | O_RDWR, 0666);
    if (g_lockFd == -1) {
        std::fprintf(stderr, "ChronoSync: [tryAcquireGlobalLock] CRITICAL - open failed (errno: %s)\n", std::strerror(errno));
        return false;
    }

    if (flock(g_lockFd, LOCK_EX | LOCK_NB) == -1) {
        if (errno == EWOULDBLOCK) {
            std::fprintf(stderr, "ChronoSync: [tryAcquireGlobalLock] WARNING - Singleton already running\n");
        } else {
            std::fprintf(stderr, "ChronoSync: [tryAcquireGlobalLock] CRITICAL - flock failed (errno: %s)\n", std::strerror(errno));
        }
        close(g_lockFd);
        g_lockFd = -1;
        return false;
    }

    std::fprintf(stderr, "ChronoSync: [tryAcquireGlobalLock] INFO - Lock acquired (PID: %d)\n", getpid());
    return true;
}

void ChronoSyncDataSource::releaseGlobalLock() {
    if (g_lockFd != -1) {
        flock(g_lockFd, LOCK_UN);
        close(g_lockFd);
        unlink(LOCK_FILE_PATH);
        g_lockFd = -1;
        std::fprintf(stderr, "ChronoSync: [releaseGlobalLock] INFO - Lock released (PID: %d)\n", getpid());
    }
}

void ChronoSyncDataSource::init() {
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] INFO - Called (PID: %d)\n", getpid());

    if (!tryAcquireGlobalLock()) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] WARNING - Another instance active\n");
        return;
    }

    if (m_private != nullptr) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] WARNING - Already initialized\n");
        return;
    }

    m_private = new ChronoSyncDataSourcePrivate(this);
    m_private->m_workerRunning = true;
    m_private->m_worker = new std::thread(&ChronoSyncDataSourcePrivate::work, m_private);

    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] INFO - Worker thread created\n");
    std::fprintf(stderr,
                 "ChronoSync: [ChronoSyncDataSource::init] DEBUG - Sync clock offset: %ld drift: %ld last_host_ts: %ld\n",
                 chrono_sync::offset.load(),
                 chrono_sync::drift.load(),
                 chrono_sync::last_host_timestamp.load());
}

void ChronoSyncDataSource::startTracing() {
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] INFO - Called (PID: %d)\n", getpid());

    if (m_private == nullptr) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] WARNING - Not singleton instance\n");
        return;
    }

    if (m_workExecuted) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] WARNING - Work already executed\n");
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_wait.notify_all();
}

void ChronoSyncDataSource::stopTracing() {
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::stopTracing] INFO - Called (PID: %d)\n", getpid());
    if (m_private == nullptr) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::stopTracing] WARNING - Not singleton instance\n");
    }
}

void ChronoSyncDataSource::flush() {
    if (m_private == nullptr) {
        return;
    }

    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::flush] INFO - Called\n");
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_private->m_workerRunning) {
        m_wait.wait(lock);
    }
}

void ChronoSyncDataSource::end() {
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::end] INFO - Called (PID: %d)\n", getpid());

    if (m_private == nullptr) {
        releaseGlobalLock();
        return;
    }

    if (m_private->m_worker != nullptr) {
        m_private->m_done.store(true, std::memory_order_relaxed);
        m_wait.notify_one();
        m_private->m_worker->join();
        delete m_private->m_worker;
        m_private->m_worker = nullptr;
    }

    releaseGlobalLock();
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::end] INFO - Cleanup complete\n");
}

// -----------------------------------------------------------------------------
// Work Routine
// -----------------------------------------------------------------------------
void ChronoSyncDataSource::work() {
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Starting\n");
    ++m_messageCount;

    auto now = std::chrono::system_clock::now();
    const std::time_t nowSeconds = std::chrono::system_clock::to_time_t(now);
    auto nowMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char timeBuffer[26];
    ctime_r(&nowSeconds, timeBuffer);
    timeBuffer[24] = '\0';
    std::fprintf(stderr,
                 "[%s.%03lld] ChronoSync Work #%d: Performing synchronization...\n",
                 timeBuffer,
                 static_cast<long long>(nowMilliseconds.count()),
                 m_messageCount);

    const char* configPath = std::getenv("RPDT_CLOCKSYNC_IP");
    if (configPath == nullptr) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] WARNING - RPDT_CLOCKSYNC_IP not set\n");
        return;
    }

    FILE* fileHandle = std::fopen(configPath, "r");
    if (fileHandle == nullptr) {
        std::fprintf(stderr,
                     "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - Failed to open %s (errno: %s)\n",
                     configPath,
                     std::strerror(errno));
        return;
    }

    std::string hostIp;
    int hostRank = -1;
    std::vector<std::pair<std::string, int>> neighbors;
    char lineBuffer[256];

    const char* myRankEnv = std::getenv("RPDT_CLOCKSYNC_RANK");
    int myRank = myRankEnv ? std::stoi(myRankEnv) : -1;

    while (std::fgets(lineBuffer, sizeof(lineBuffer), fileHandle)) {
        std::string line(lineBuffer);
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty()) continue;

        // Parse: "10.7.76.147,rank=0"
        size_t commaPos = line.find(',');
        if (commaPos == std::string::npos) continue;
        
        std::string ip = line.substr(0, commaPos);
        std::string rankStr = line.substr(commaPos + 1);
        
        // Extract rank value from "rank=0"
        if (rankStr.substr(0, 5) != "rank=") continue;
        
        int fileRank = std::stoi(rankStr.substr(5));
        
        if (myRank == fileRank) {
            hostRank = fileRank;
            hostIp = ip;
        } else {
            neighbors.emplace_back(ip, fileRank);
        }
    }

    std::fclose(fileHandle);
    // print parsed values
    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Host IP: %s, Host Rank: %d\n", hostIp.c_str(), hostRank);
    for (const auto& neighbor : neighbors) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Neighbor IP: %s, Neighbor Rank: %d\n", neighbor.first.c_str(), neighbor.second);
    }

    if (m_private == nullptr) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - Private data missing\n");
        return;
    }

    m_private->m_hostIp = hostIp;
    m_private->m_rank = hostRank;
    m_private->m_neighbors = neighbors;

    firefly::SharedMemoryRegion sharedRegion;
    if (!sharedRegion.initialize(firefly::MAX_MEASUREMENT_COUNT)) {
        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - Shared memory init failed\n");
        return;
    }

    std::vector<pid_t> childPids;
    for (const auto& neighbor : neighbors) {
        pid_t forkResult = fork();
        if (forkResult == 0) {
            if (hostRank < neighbor.second) {
                // print my host rank
                printf("My host rank is %d\n", hostRank);
                printf("Neighbor host rank is %d\n", neighbor.second);
                firefly::run_node("A",
                         hostIp.c_str(),
                         neighbor.first.c_str(),
                         firefly::UDP_PORT_DEFAULT + hostRank + neighbor.second, // TODO FOR ALI may need to make it rank id based
                         sharedRegion.measurements(),
                         sharedRegion.measurementCount(),
                         sharedRegion.semaphore());
            } else {
                printf("My host rank is %d\n", hostRank);
                printf("Neighbor host rank is %d\n", neighbor.second);
                firefly::run_node("B",
                         neighbor.first.c_str(),
                         hostIp.c_str(),
                         firefly::UDP_PORT_DEFAULT + hostRank + neighbor.second, // TODO FOR ALI may need to make it rank id based
                         sharedRegion.measurements(),
                         sharedRegion.measurementCount(),
                         sharedRegion.semaphore());
            }
            std::exit(EXIT_SUCCESS);
        } else if (forkResult > 0) {
            childPids.push_back(forkResult);
        } else {
            std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - fork failed (errno: %s)\n", std::strerror(errno));
        }
    }

    while (!m_private->m_done.load(std::memory_order_relaxed)) {
        if (!neighbors.empty()) {
            const char* role = (hostRank < neighbors.front().second) ? "A" : "B";
            firefly::firefly_run(role,
                        sharedRegion.measurements(),
                        sharedRegion.measurementCount(),
                        sharedRegion.semaphore());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(firefly::FIRE_FLY_SLEEP_MSEC));
    }

    for (pid_t pidValue : childPids) {
        kill(pidValue, SIGTERM);
        waitpid(pidValue, nullptr, 0);
    }

    // std::fprintf(stderr,
    //              "ChronoSync: [ChronoSyncDataSource::work] INFO - Collected %d measurements\n",
    //              *sharedRegion.measurementCount());

    // for (int index = 0; index < *sharedRegion.measurementCount(); ++index) {
    //     const firefly::Measurement& measurement = sharedRegion.measurements()[index];
    //     std::fprintf(stderr,
    //                  "Measurement %d: Node %s, Timestamp: %lld, t_a: %lld, r_a: %lld, t_b: %lld, r_b: %lld, RTT: %lld, Offset: %lld, UDP Port: %d\n",
    //                  index,
    //                  measurement.node,
    //                  static_cast<long long>(measurement.timestampNs),
    //                  static_cast<long long>(measurement.sendTimeA),
    //                  static_cast<long long>(measurement.recvTimeA),
    //                  static_cast<long long>(measurement.sendTimeB),
    //                  static_cast<long long>(measurement.recvTimeB),
    //                  static_cast<long long>(measurement.roundTripTime),
    //                  measurement.offset,
    //                  measurement.udpPort);
    // }

    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Synchronization complete\n");
}
