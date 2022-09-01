/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"
#include "ApiIdList.h"

class CuptiDataSource : public DataSource
{
public:
    CuptiDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;



private:
    ApiIdList m_apiList {nullptr};

    //static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
    //static void activity_callback(const char* begin, const char* end, void* arg);


};
