/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "SmuDumpDataSource.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include <fmt/format.h>

#include "Logger.h"

static SmuDumpDataSource *g_singleton = nullptr;

// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *SmuDumpDataSourceFactory() { return (g_singleton = new SmuDumpDataSource()); }
}  // extern "C"


SmuDumpDataSource& SmuDumpDataSource::singleton()
{
    return *g_singleton;
}

timestamp_t SmuDumpDataSource::getTimeStamp()
{
    return m_timestamp;
}

bool SmuDumpDataSource::isLoggingEnabled()
{
    return m_loggingEnabled;
}

void SmuDumpDataSource::init()
{
    void (*dl) = dlopen("libsmutrace.so", RTLD_LAZY);
    if (dl) {
        f_smuDumpInit = (SmuDumpInitFunc) dlsym(dl, "smuDumpInit");
        f_smuDumpEnd = (SmuDumpEndFunc) dlsym(dl, "smuDumpEnd");
        f_smuDumpOnce = (SmuDumpOnceFunc) dlsym(dl, "smuDumpOnce");
        m_loggingEnabled = (f_smuDumpInit && f_smuDumpEnd && f_smuDumpOnce &&
                            f_smuDumpInit(addSMUValueToSqliteDb));
    }

    // FIXME: decide how many gpus and what values to log

    m_done = false;
    m_period = 1000;
    m_timestamp=0;

    if (m_loggingEnabled)
    {
        m_resource = new DbResource(Logger::singleton().filename(), std::string("smudump_logger_active"));
        m_worker = new std::thread(&SmuDumpDataSource::work, this);
    }
}

void SmuDumpDataSource::end()
{
    if (m_loggingEnabled)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done = true;
        lock.unlock();
        m_worker->join();
        delete m_worker;
        m_resource->unlock();
        f_smuDumpEnd();
    }
}

void SmuDumpDataSource::startTracing()
{
    if (m_loggingEnabled)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_loggingActive = true;
    }
}

void SmuDumpDataSource::stopTracing()
{
    if (m_loggingEnabled)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_timestamp = 0;
        m_loggingActive = false;

        // Tell the monitor table that it should terminate any outstanding ranges...
        //    since we are paused/stopped.
        Logger &logger = Logger::singleton();
        logger.monitorTable().endCurrentRuns(clocktime_ns());
    }
}


void SmuDumpDataSource::addSMUValueToSqliteDb(uint64_t did, const char* type ,const char* name, double value)
{
    if (SmuDumpDataSource::singleton().getTimeStamp() == 0) return;
    Logger &logger = Logger::singleton();
    MonitorTable::row mrow;
    mrow.deviceId = did;
    mrow.deviceType = type;
    mrow.monitorType = name;
    mrow.start = SmuDumpDataSource::singleton().getTimeStamp() - 1000000;
    mrow.end = 0;
    mrow.value = fmt::format("{}", value);
    logger.monitorTable().insert(mrow);
}

void SmuDumpDataSource::work()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    sqlite3_int64 startTime = clocktime_ns()/1000;

    bool haveResource = m_resource->tryLock();
    
    while (m_done == false) {

        if (haveResource && m_loggingActive) {
            lock.unlock();
            m_timestamp=clocktime_ns(); //single timestamp for all variables dumped at once, for easier post-processing.
            f_smuDumpOnce();
            lock.lock();
        }
        
        sqlite3_int64 sleepTime = startTime + m_period - clocktime_ns()/1000;
        sleepTime = (sleepTime > 0) ? sleepTime : 0;
        if (haveResource == false)
            sleepTime += m_period * 10;

        lock.unlock();
        usleep(sleepTime);
        lock.lock();
        if (haveResource == false) {
            haveResource = m_resource->tryLock();
        }
        startTime = clocktime_ns()/1000;
    }
}

