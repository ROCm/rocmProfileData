/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"
#include "DbResource.h"
#include "Utility.h"

#include <sqlite3.h>
#include <thread>
#include <mutex>
#include <condition_variable>

typedef void (*SmuDumpCallback)(uint64_t, const char*, const char*, double);
typedef bool (*SmuDumpInitFunc) (SmuDumpCallback callback);
typedef void (*SmuDumpEndFunc) (void);
typedef void (*SmuDumpOnceFunc) (void);
typedef void (*RegDumpOnceFunc) (void);
typedef uint32_t (*RegGetTraceRate) (void);
typedef uint32_t (*SmuGetTraceRate) (void);


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
    bool isLoggingEnabled();

private:
    std::mutex m_mutex;
    SmuDumpInitFunc f_smuDumpInit;
    SmuDumpEndFunc f_smuDumpEnd;
    SmuDumpOnceFunc f_smuDumpOnce;
    RegDumpOnceFunc f_regDumpOnce;
    RegGetTraceRate f_regGetTraceRate;
    SmuGetTraceRate f_smuGetTraceRate;
    DbResource *m_smu_resource {nullptr};
    DbResource *m_reg_resource {nullptr};
    timestamp_t m_timestamp;

    bool m_loggingActive {false};
    bool m_loggingEnabled {false};
    static void addSMUValueToSqliteDb(uint64_t, const char* type ,const char* name, double value);


    void smuwork(); 
    void regwork();                
    std::thread *m_smu_worker {nullptr};
    std::thread *m_reg_worker {nullptr};
    volatile bool m_done {false};
    sqlite3_int64 m_smu_period { 1000 };
    sqlite3_int64 m_reg_period { 1000 };
};

