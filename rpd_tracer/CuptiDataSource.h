/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once 

#include <cuda.h>
#include <cupti.h>

#include "DataSource.h"
#include "ApiIdList.h"

class CudaApiIdList : public ApiIdList
{
public:
    CudaApiIdList();
    uint32_t mapName(const std::string &apiName) override;
private:
    std::unordered_map<std::string, uint32_t> m_nameMap;
};

class CuptiDataSource : public DataSource
{
public:
    //CuptiDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;



private:
    CudaApiIdList m_apiList;

    CUpti_SubscriberHandle m_subscriber;

    static void CUPTIAPI api_callback(void *userdata, CUpti_CallbackDomain domain,
		                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);

    static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

};
