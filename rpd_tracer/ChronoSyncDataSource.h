/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#ifndef CHRONOSYNCDATASOURCE_H
#define CHRONOSYNCDATASOURCE_H

#include "DataSource.h"

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

private:
    void storeMetadata(const std::string& tag, const std::string& value);
    std::string queryMetadata(const std::string& tag);

    ChronoSyncDataSourcePrivate* m_private;
    DbResource* m_resource;
    std::string m_shmName;

    friend class ChronoSyncDataSourcePrivate;
};

}    // namespace rpdtracer

#endif // CHRONOSYNCDATASOURCE_H
