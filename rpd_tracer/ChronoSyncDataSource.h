/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#ifndef CHRONOSYNCDATASOURCE_H
#define CHRONOSYNCDATASOURCE_H

#include "DataSource.h"

#include <condition_variable>
#include <mutex>

// Forward declaration
class ChronoSyncDataSourcePrivate;

class ChronoSyncDataSource : public DataSource
{
public:
    ChronoSyncDataSource();
    ~ChronoSyncDataSource();  // REMOVED 'override' - base class destructor is not virtual

    void init() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;
    void end() override;

    void work();

    // Public synchronization primitives accessed by the worker
    std::mutex m_mutex;
    std::condition_variable m_wait;
    bool m_workExecuted;  // MOVED to public for friend class access

private:
    bool tryAcquireGlobalLock();
    void releaseGlobalLock();

    ChronoSyncDataSourcePrivate* m_private;
    int m_messageCount;

    friend class ChronoSyncDataSourcePrivate;
};

#endif // CHRONOSYNCDATASOURCE_H