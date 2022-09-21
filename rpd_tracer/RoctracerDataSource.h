/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"
#include "ApiIdList.h"

class RocmApiIdList : public ApiIdList
{
public:
    RocmApiIdList() { ; }
    uint32_t mapName(const std::string &apiName) override;
};


class RoctracerDataSource : public DataSource
{
public:
    //RoctracerDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;

private:
    RocmApiIdList m_apiList;

    roctracer_pool_t *m_hccPool{nullptr};
    static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
    static void hcc_activity_callback(const char* begin, const char* end, void* arg);

};
