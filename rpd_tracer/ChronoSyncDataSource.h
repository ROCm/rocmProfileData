/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#ifndef CHRONOSYNCDATASOURCE_H
#define CHRONOSYNCDATASOURCE_H

#include "DataSource.h"

#include <condition_variable>
#include <mutex>
#include <string>

namespace rpdtracer {

// Forward declarations
class ChronoSyncDataSourcePrivate;
class DbResource;

class ChronoSyncDataSource : public DataSource
{
public:
    ChronoSyncDataSource();
    ~ChronoSyncDataSource();

    void init() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;
    void end() override;

    void work();

    std::mutex m_mutex;
    std::condition_variable m_wait;
    bool m_workExecuted;

private:
    void storeMetadata(const std::string& tag, const std::string& value);
    std::string queryMetadata(const std::string& tag);

    ChronoSyncDataSourcePrivate* m_private;
    DbResource* m_resource;
    std::string m_shmName;
    int m_messageCount;

    friend class ChronoSyncDataSourcePrivate;
};

}    // namespace rpdtracer

#endif // CHRONOSYNCDATASOURCE_H
