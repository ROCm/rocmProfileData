/*********************************************************************************
* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
********************************************************************************/
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
