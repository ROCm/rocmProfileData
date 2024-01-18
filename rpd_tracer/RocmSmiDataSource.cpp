/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "RocmSmiDataSource.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "rocm_smi/rocm_smi.h"

#include <fmt/format.h>

#include "Logger.h"
#include "Utility.h"


// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *RocmSmiDataSourceFactory() { return new RocmSmiDataSource(); }
}  // extern "C"



void RocmSmiDataSource::init()
{
    rsmi_status_t ret;
    ret = rsmi_init(0);

#if 0
    uint32_t num_devices;
    uint16_t dev_id;

    rsmi_num_monitor_devices(&num_devices);
    for (int i = 0; i < num_devices; ++i) {
        rsmi_dev_id_get(i, &dev_id);
        fprintf(stderr, "device: %d\n", dev_id);
    }
#endif

    m_done = false;
    m_period = 1000;

    m_resource = new DbResource(Logger::singleton().filename(), std::string("smi_logger_active"));
    m_worker = new std::thread(&RocmSmiDataSource::work, this);
}

void RocmSmiDataSource::end()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_done = true;
    lock.unlock();
    m_worker->join();
    delete m_worker;

    m_resource->unlock();

    rsmi_status_t ret;
    ret = rsmi_shut_down();
}

void RocmSmiDataSource::startTracing()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_loggingActive = true;
}

void RocmSmiDataSource::stopTracing()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_loggingActive = false;

    // Tell the monitor table that it should terminate any outstanding ranges...
    //    since we are paused/stopped.
    Logger &logger = Logger::singleton();
    logger.monitorTable().endCurrentRuns(clocktime_ns());
}

void RocmSmiDataSource::flush() {
    Logger &logger = Logger::singleton();
    logger.monitorTable().endCurrentRuns(clocktime_ns());
}


void RocmSmiDataSource::work()
{
    Logger &logger = Logger::singleton();
    std::unique_lock<std::mutex> lock(m_mutex);

    sqlite3_int64 startTime = clocktime_ns()/1000;

    bool haveResource = m_resource->tryLock();
    
    while (m_done == false) {
        if (haveResource && m_loggingActive) {
            lock.unlock();

            uint32_t num_devices = 1;
            uint16_t dev_id = 0;

            rsmi_num_monitor_devices(&num_devices);
            for (int i = 0; i < num_devices; ++i) {
                rsmi_status_t ret;

#if 1
                rsmi_frequencies_t freqs;
                ret = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_SYS, &freqs);
                if (ret == RSMI_STATUS_SUCCESS) {
                    MonitorTable::row mrow;
                    mrow.deviceId = i;
                    mrow.deviceType = "gpu";	// FIXME, use enums or somthing fancy
                    mrow.monitorType = "sclk";	// FIXME, use enums or somthing fancy
                    mrow.start = clocktime_ns();
                    mrow.end = 0;
                    mrow.value = fmt::format("{}", freqs.frequency[freqs.current] / 1000000);
                    logger.monitorTable().insert(mrow);
                }
#endif
#if 0
                uint64_t pow;
                ret = rsmi_dev_power_ave_get(i, 0, &pow);
                if (ret == RSMI_STATUS_SUCCESS) {
                    MonitorTable::row mrow;
                    mrow.deviceId = i;
                    mrow.deviceType = "gpu";	// FIXME, use enums or somthing fancy
                    mrow.monitorType = "power";	// FIXME, use enums or somthing fancy
                    mrow.start = clocktime_ns();
                    mrow.end = 0;
                    mrow.value = fmt::format("{}", pow / 1000000.0);
                    logger.monitorTable().insert(mrow);
                }
#endif
#if 0
                int64_t temp;
                ret = rsmi_dev_temp_metric_get(i, RSMI_TEMP_TYPE_FIRST, RSMI_TEMP_CURRENT, &temp);
                if (ret == RSMI_STATUS_SUCCESS) {
                    MonitorTable::row mrow;
                    mrow.deviceId = i;
                    mrow.deviceType = "gpu";	// FIXME, use enums or somthing fancy
                    mrow.monitorType = "temp";	// FIXME, use enums or somthing fancy
                    mrow.start = clocktime_ns();
                    mrow.end = 0;
                    mrow.value = fmt::format("{}", temp/1000);
                    logger.monitorTable().insert(mrow);
                }
#endif
            }
            lock.lock();
        }
        
        sqlite3_int64 sleepTime = startTime + m_period - clocktime_ns()/1000;
        sleepTime = (sleepTime > 0) ? sleepTime : 0;
        // sleep longer if we aren't the active instance
        if (haveResource == false)
            sleepTime += m_period * 10;
        lock.unlock();
        usleep(sleepTime);
        lock.lock();
        // Try to become the active logging instance
        if (haveResource == false) {
            haveResource = m_resource->tryLock();
        }
        startTime = clocktime_ns()/1000;
    }
}
