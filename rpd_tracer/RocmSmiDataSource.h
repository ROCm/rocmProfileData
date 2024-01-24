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




class RocmSmiDataSource : public DataSource
{
public:
    //RoctracerDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    virtual void flush() override;

private:
    std::mutex m_mutex;
    std::condition_variable m_wait;
    bool m_loggingActive {false};
    DbResource *m_resource {nullptr};

    void work();                // work thread
    std::thread *m_worker {nullptr};
    volatile bool m_done {false};
    bool m_workerRunning {false};
    sqlite3_int64 m_period { 10000 };
};

