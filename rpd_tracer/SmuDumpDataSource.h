/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"
#include "DbResource.h"

#include <sqlite3.h>
#include <thread>
#include <mutex>
#include <condition_variable>

typedef void (*SmuDumpCallback)(uint64_t, const char*, const char*, double);
typedef bool (*SmuDumpInitFunc) (SmuDumpCallback callback);
typedef void (*SmuDumpEndFunc) (void);
typedef void (*SmuDumpOnceFunc) (void);

class SmuDumpDataSource : public DataSource
{
public:
    //RoctracerDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    static SmuDumpDataSource& singleton();
    timestamp_t getTimeStamp();

private:
    std::mutex m_mutex;
    SmuDumpInitFunc f_smuDumpInit;
    SmuDumpEndFunc f_smuDumpEnd;
    SmuDumpOnceFunc f_smuDumpOnce;
    DbResource *m_resource {nullptr};
    static SmuDumpDataSource *m_singleton;
    timestamp_t m_timestamp;

    bool m_loggingActive {false};
    bool m_loggingEnabled {false};
    static void addSMUValueToSqliteDb(uint64_t, const char* type ,const char* name, double value);


    void work();                // work thread
    std::thread *m_worker {nullptr};
    volatile bool m_done {false};
    sqlite3_int64 m_period { 1000 };
};

