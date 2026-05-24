/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ChronoSyncDataSource.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <string>
#include <thread>

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
        if (m_owner != nullptr)
            m_owner->work();
    }

    ChronoSyncDataSource* m_owner{nullptr};
    std::thread* m_worker{nullptr};
    std::atomic<bool> m_done{false};
};

// -----------------------------------------------------------------------------
// ChronoSyncDataSource Implementation
// -----------------------------------------------------------------------------
ChronoSyncDataSource::ChronoSyncDataSource()
    : m_private(nullptr),
      m_resource(nullptr) {}

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
    if (m_private != nullptr)
        return;

    m_resource = new DbResource(rpd_filename(), std::string("chronosync_active"));
    if (!m_resource->tryLock()) {
        // Not the delegate — attach to existing shared memory and block until synced
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(firefly::SYNC_TIMEOUT_SEC);

        std::string existingShm;
        while (std::chrono::steady_clock::now() < deadline) {
            existingShm = queryMetadata("clocksync_shm");
            if (!existingShm.empty())
                break;
            usleep(100000);
        }

        if (existingShm.empty()) {
            std::fprintf(stderr, "ChronoSync: WARNING - shared memory not available after %ds\n",
                         firefly::SYNC_TIMEOUT_SEC);
            return;
        }

        firefly::attach_clocksync_shm(existingShm);

        while (std::chrono::steady_clock::now() < deadline) {
            if (firefly::g_pSvcState->sequence.load(std::memory_order_acquire) > 0)
                return;
            usleep(100000);
        }

        std::fprintf(stderr, "ChronoSync: WARNING - clock sync not available after %ds, "
                     "proceeding with uncorrected timestamps\n", firefly::SYNC_TIMEOUT_SEC);
        return;
    }

    firefly::create_clocksync_shm(m_shmName);
    storeMetadata("clocksync_shm", m_shmName);

    m_private = new ChronoSyncDataSourcePrivate(this);
    m_private->m_worker = new std::thread(&ChronoSyncDataSourcePrivate::work, m_private);
}

void ChronoSyncDataSource::startTracing() {
}

void ChronoSyncDataSource::stopTracing() {
}

void ChronoSyncDataSource::flush() {
}

void ChronoSyncDataSource::end() {
    if (m_private == nullptr)
        return;

    if (m_private->m_worker != nullptr) {
        m_private->m_done.store(true, std::memory_order_relaxed);
        m_private->m_worker->join();
        delete m_private->m_worker;
        m_private->m_worker = nullptr;
    }

    if (m_resource != nullptr)
        m_resource->unlock();

    if (!m_shmName.empty())
        firefly::cleanup_clocksync_shm(m_shmName);
}

// -----------------------------------------------------------------------------
// Work Routine (hub-and-spoke)
// -----------------------------------------------------------------------------
void ChronoSyncDataSource::work() {
    int rank = -1;
    std::string masterAddr;

    // Config fallback chain:
    // 1. torchrun env vars
    const char* groupRank = std::getenv("GROUP_RANK");
    const char* masterAddrEnv = std::getenv("MASTER_ADDR");
    if (groupRank != nullptr && masterAddrEnv != nullptr) {
        rank = std::atoi(groupRank);
        masterAddr = masterAddrEnv;
    }

    // 2. standalone env vars
    if (rank < 0) {
        const char* rpdtRank = std::getenv("RPDT_CLOCKSYNC_RANK");
        const char* rpdtMaster = std::getenv("RPDT_CLOCKSYNC_MASTER");
        if (rpdtRank != nullptr)
            rank = std::atoi(rpdtRank);
        if (rpdtMaster != nullptr)
            masterAddr = rpdtMaster;
    }

    // 3. no config — no clock sync
    if (rank < 0)
        return;

    int port = firefly::CLOCKSYNC_PORT_DEFAULT;
    const char* portEnv = std::getenv("RPDT_CLOCKSYNC_PORT");
    if (portEnv != nullptr)
        port = std::atoi(portEnv);

    firefly::MeasurementBuffer buffer(firefly::MAX_MEASUREMENT_COUNT);
    std::atomic<bool> done{false};

    if (rank == 0) {
        // Server: respond to probes from all clients. Offset stays 0.
        std::thread serverThread([&]() {
            firefly::run_server(port, buffer, done);
        });

        while (!m_private->m_done.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(firefly::FIRE_FLY_SLEEP_MSEC));

        done.store(true, std::memory_order_relaxed);
        serverThread.join();
    } else {
        // Client: connect to server and measure offset
        std::thread clientThread([&]() {
            firefly::run_client(masterAddr.c_str(), port, buffer, done);
        });

        int lastCount = 0;
        while (!m_private->m_done.load(std::memory_order_relaxed)) {
            if (buffer.count != lastCount) {
                lastCount = buffer.count;
                firefly::firefly_run("B", buffer);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(firefly::FIRE_FLY_SLEEP_MSEC));
        }

        done.store(true, std::memory_order_relaxed);
        clientThread.join();
    }
}

}    // namespace rpdtracer
