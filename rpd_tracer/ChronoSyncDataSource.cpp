/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ChronoSyncDataSource.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "DbResource.h"
#include "Logger.h"
#include "Utility.h"
#include "Firefly.h"

using rpdtracer::DataSource;
using rpdtracer::ChronoSyncDataSource;

// Create a factory for the Logger to locate and use
extern "C" {
DataSource* ChronoSyncDataSourceFactory() {
    return new ChronoSyncDataSource();
}
}

namespace rpdtracer {

// -----------------------------------------------------------------------------
// ChronoSyncDataSourcePrivate
// -----------------------------------------------------------------------------
class ChronoSyncDataSourcePrivate {
public:
    explicit ChronoSyncDataSourcePrivate(ChronoSyncDataSource* owner)
        : m_owner(owner) {}

    void work() {
        if (m_owner == nullptr) {
//            std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSourcePrivate::work] CRITICAL - Owner not set\n");
            return;
        }

        std::unique_lock<std::mutex> lock(m_owner->m_mutex);
        while (!m_done.load(std::memory_order_relaxed)) {
            if (!m_owner->m_workExecuted) {
//                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Executing work once\n");
                m_workerRunning = true;
                lock.unlock();
                m_owner->work();
                lock.lock();
                m_owner->m_workExecuted = true;
                m_workerRunning = false;
//                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Work completed\n");
            }

            m_workerRunning = false;
            if (!m_done.load(std::memory_order_relaxed)) {
//                std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Waiting for signal\n");
                m_owner->m_wait.wait(lock);
            }
            m_workerRunning = true;
        }

//        std::fprintf(stderr, "ChronoSync: [ChronoSync Worker] INFO - Exiting\n");
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
ChronoSyncDataSource::ChronoSyncDataSource()
    : m_private(nullptr),
      m_resource(nullptr),
      m_workExecuted(false),
      m_messageCount(0) {}

ChronoSyncDataSource::~ChronoSyncDataSource() {
    end();
    delete m_private;
    m_private = nullptr;
    delete m_resource;
    m_resource = nullptr;
}

static std::string rpd_filename()
{
    const char *f = getenv("RPDT_FILENAME");
    return (f != nullptr) ? f : "./trace.rpd";
}

static int metadata_callback(void *data, int argc, char **argv, char **colName)
{
    std::string &value = *static_cast<std::string*>(data);
    if (argc > 0 && argv[0])
        value = argv[0];
    return 0;
}

void ChronoSyncDataSource::storeMetadata(const std::string& tag, const std::string& value)
{
    sqlite3 *db = nullptr;
    if (sqlite3_open(rpd_filename().c_str(), &db) != SQLITE_OK)
        return;
    sqlite3_busy_handler(db, &sqlite_busy_handler, NULL);
    char *err = nullptr;
    std::string sql = "INSERT OR REPLACE INTO rocpd_metadata(tag, value) VALUES ('" + tag + "', '" + value + "')";
    sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    sqlite3_close(db);
}

std::string ChronoSyncDataSource::queryMetadata(const std::string& tag)
{
    std::string value;
    sqlite3 *db = nullptr;
    if (sqlite3_open(rpd_filename().c_str(), &db) != SQLITE_OK)
        return value;
    sqlite3_busy_handler(db, &sqlite_busy_handler, NULL);
    char *err = nullptr;
    std::string sql = "SELECT value FROM rocpd_metadata WHERE tag = '" + tag + "'";
    sqlite3_exec(db, sql.c_str(), metadata_callback, &value, &err);
    sqlite3_close(db);
    return value;
}

void ChronoSyncDataSource::init() {
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] INFO - Called (PID: %d)\n", getpid());

    if (m_private != nullptr) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] WARNING - Already initialized\n");
        return;
    }

    m_resource = new DbResource(rpd_filename(), std::string("chronosync_active"));
    if (!m_resource->tryLock()) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] WARNING - Another instance active\n");
        // Try to attach to existing shared memory from the singleton
        std::string existingShm = queryMetadata("clocksync_shm");
        if (!existingShm.empty())
            firefly::attach_clocksync_shm(existingShm);
        return;
    }

//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] INFO - Lock acquired (PID: %d)\n", getpid());

    firefly::create_clocksync_shm(m_shmName);
    storeMetadata("clocksync_shm", m_shmName);

    m_private = new ChronoSyncDataSourcePrivate(this);
    m_private->m_workerRunning = true;
    m_private->m_worker = new std::thread(&ChronoSyncDataSourcePrivate::work, m_private);

//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::init] INFO - Worker thread created\n");
}

void ChronoSyncDataSource::startTracing() {
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] INFO - Called (PID: %d)\n", getpid());

    if (m_private == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] WARNING - Not singleton instance\n");
        return;
    }

    if (m_workExecuted) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::startTracing] WARNING - Work already executed\n");
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_wait.notify_all();
}

void ChronoSyncDataSource::stopTracing() {
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::stopTracing] INFO - Called (PID: %d)\n", getpid());
    if (m_private == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::stopTracing] WARNING - Not singleton instance\n");
    }
}

void ChronoSyncDataSource::flush() {
    if (m_private == nullptr) {
        return;
    }

//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::flush] INFO - Called\n");
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_private->m_workerRunning) {
        m_wait.wait(lock);
    }
}

void ChronoSyncDataSource::end() {
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::end] INFO - Called (PID: %d)\n", getpid());

    if (m_private == nullptr)
        return;

    if (m_private->m_worker != nullptr) {
        m_private->m_done.store(true, std::memory_order_relaxed);
        m_wait.notify_one();
        m_private->m_worker->join();
        delete m_private->m_worker;
        m_private->m_worker = nullptr;
    }

    if (m_resource != nullptr)
        m_resource->unlock();

    if (!m_shmName.empty())
        firefly::cleanup_clocksync_shm(m_shmName);

//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::end] INFO - Cleanup complete\n");
}

// -----------------------------------------------------------------------------
// Work Routine
// -----------------------------------------------------------------------------
void ChronoSyncDataSource::work() {
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Starting\n");
    ++m_messageCount;

    auto now = std::chrono::system_clock::now();
    const std::time_t nowSeconds = std::chrono::system_clock::to_time_t(now);
    auto nowMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char timeBuffer[26];
    ctime_r(&nowSeconds, timeBuffer);
    timeBuffer[24] = '\0';
//    std::fprintf(stderr,
//                 "[%s.%03lld] ChronoSync Work #%d: Performing synchronization...\n",
//                 timeBuffer,
//                 static_cast<long long>(nowMilliseconds.count()),
//                 m_messageCount);

    const char* configPath = std::getenv("RPDT_CLOCKSYNC_IP");
    if (configPath == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] WARNING - RPDT_CLOCKSYNC_IP not set\n");
        return;
    }

    FILE* fileHandle = std::fopen(configPath, "r");
    if (fileHandle == nullptr) {
//        std::fprintf(stderr,
//                     "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - Failed to open %s (errno: %s)\n",
//                     configPath,
//                     std::strerror(errno));
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
//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Host IP: %s, Host Rank: %d\n", hostIp.c_str(), hostRank);
    for (const auto& neighbor : neighbors) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Neighbor IP: %s, Neighbor Rank: %d\n", neighbor.first.c_str(), neighbor.second);
    }

    if (m_private == nullptr) {
//        std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] CRITICAL - Private data missing\n");
        return;
    }

    m_private->m_hostIp = hostIp;
    m_private->m_rank = hostRank;
    m_private->m_neighbors = neighbors;

    firefly::MeasurementBuffer buffer(firefly::MAX_MEASUREMENT_COUNT);
    bool done = false;

    std::vector<std::thread> workers;
    for (const auto& neighbor : neighbors) {
        const char* role = (hostRank < neighbor.second) ? "A" : "B";
        const char* nodeIpA = (hostRank < neighbor.second) ? hostIp.c_str() : neighbor.first.c_str();
        const char* nodeIpB = (hostRank < neighbor.second) ? neighbor.first.c_str() : hostIp.c_str();
        int udpPort = firefly::UDP_PORT_DEFAULT + hostRank + neighbor.second;

        std::string ipACopy(nodeIpA);
        std::string ipBCopy(nodeIpB);
        std::string roleCopy(role);

        workers.emplace_back([ipACopy, ipBCopy, roleCopy, udpPort, &buffer, &done]() {
            firefly::run_node(roleCopy.c_str(), ipACopy.c_str(), ipBCopy.c_str(),
                              udpPort, buffer, done);
        });
    }

    int lastCount = 0;
    while (!m_private->m_done.load(std::memory_order_relaxed)) {
        if (!neighbors.empty() && buffer.count != lastCount) {
            lastCount = buffer.count;
            const char* role = (hostRank < neighbors.front().second) ? "A" : "B";
            firefly::firefly_run(role, buffer);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(firefly::FIRE_FLY_SLEEP_MSEC));
    }

    done = true;
    for (auto& worker : workers)
        worker.join();

//    std::fprintf(stderr, "ChronoSync: [ChronoSyncDataSource::work] INFO - Synchronization complete\n");
}

}    // namespace rpdtracer
