/*********************************************************************************
* Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "CuptiDataSource.h"

#include <sqlite3.h>
#include <fmt/format.h>

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "generated_nvtx_meta.h"

#include "Logger.h"
#include "Utility.h"


// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *CuptiDataSourceFactory() { return new CuptiDataSource(); }
}  // extern "C"

// FIXME: can we avoid shutdown corruption?
// Other libraries crashing on unload
// libsqlite unloading before we are done using it
// Current workaround: register an onexit function when first activity is delivered back
//                     this let's us unload first, or close to.
// New workaround: register 3 times, only finalize once.  see register_once

static std::once_flag register_once;
static std::once_flag registerAgain_once;

void CuptiDataSource::init()
{

    // Pick some apis to ignore
    m_apiList.setInvertMode(true);  // Omit the specified api
    m_apiList.add("cudaGetDevice_v3020");
    m_apiList.add("cudaSetDevice_v3020");
    m_apiList.add("cudaGetLastError_v3020");

    //FIXME: gross
    setenv("NVTX_INJECTION64_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so", 0);

    // FIXME: cuptiSubscribe may fail with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
    cuptiSubscribe(&m_subscriber, (CUpti_CallbackFunc)api_callback, nullptr);
    cuptiActivityRegisterCallbacks(CuptiDataSource::bufferRequested, CuptiDataSource::bufferCompleted);
    //cuptiActivityRegisterTimestampCallback();	// Cuda 11.6 :(

    // Callback API
    //cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    //cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_NVTX);

    // Async Activity
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD);
}

void CuptiDataSource::end()
{
    cuptiActivityFlushAll(1);
}

void CuptiDataSource::startTracing()
{
    if (m_apiList.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
        const std::unordered_map<uint32_t, uint32_t> &filter = m_apiList.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        cuptiEnableDomain(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
        const std::unordered_map<uint32_t, uint32_t> &filter = m_apiList.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, it->first);
        }
    }

    cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_NVTX);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

void CuptiDataSource::stopTracing()
{
    cuptiEnableDomain(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableDomain(0, m_subscriber, CUPTI_CB_DOMAIN_NVTX);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

void CUPTIAPI CuptiDataSource::api_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    Logger &logger = Logger::singleton();

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        thread_local sqlite3_int64 timestamp;
        //const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;   

        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            timestamp = clocktime_ns();
        }
        else { // cbInfo->callbackSite == CUPTI_API_EXIT
            char buff[4096];
            ApiTable::row row;

            const char *name = "";
            cuptiGetCallbackName(domain, cbid, &name);
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(name);
            row.pid = GetPid();
            row.tid = GetTid();
            row.start = timestamp;  // From TLS from preceding enter call
            row.end = clocktime_ns();
            row.apiName_id = name_id;
            row.args_id = EMPTY_STRING_ID;
            row.api_id = cbInfo->correlationId;

            switch (cbid) {
                case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
                    {
                        auto &params = *(cudaMalloc_v3020_params_st *)(cbInfo->functionParams);
                        std::snprintf(buff, 4096, "ptr=%p | size=0x%x",
                            *params.devPtr,
                            (uint32_t)(params.size));
                        row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
                    {
                        auto &params = *(cudaFree_v3020_params_st *)(cbInfo->functionParams);
                        std::snprintf(buff, 4096, "ptr=%p",
                            params.devPtr);
                        row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
                    {
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
                    {
                        auto &params = *(cudaLaunchKernel_v7000_params_st *)(cbInfo->functionParams);
                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = params.gridDim.x;
                        krow.gridY = params.gridDim.y;
                        krow.gridZ = params.gridDim.z;
                        krow.workgroupX = params.blockDim.x;
                        krow.workgroupY = params.blockDim.y;
                        krow.workgroupZ = params.blockDim.z;
                        krow.groupSegmentSize = params.sharedMem;
                        krow.privateSegmentSize = 0;
                        if (cbInfo->symbolName != nullptr)  // Happens, why?  "" duh
                            krow.kernelName_id = logger.stringTable().getOrCreate(cxx_demangle(cbInfo->symbolName));
                        else
                            krow.kernelName_id = EMPTY_STRING_ID;
                        logger.kernelApiTable().insert(krow);
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
                    {
                        auto &params = *(cudaMemcpy_v3020_params *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
                    {
                        auto &params = *(cudaMemcpyAsync_v3020_params *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020:
                     {
                        auto &params = *(cudaMemcpy2D_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020:
                     {
                        auto &params = *(cudaMemcpyToArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_v3020:
                     {
                        auto &params = *(cudaMemcpy2DToArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020:
                     {
                        auto &params = *(cudaMemcpyFromArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_v3020:
                     {
                        auto &params = *(cudaMemcpy2DFromArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020:
                     {
                        auto &params = *(cudaMemcpyArrayToArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_v3020:
                     {
                        auto &params = *(cudaMemcpy2DArrayToArray_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020:
                     {
                        auto &params = *(cudaMemcpyToSymbol_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.symbol);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020:
                     {
                        auto &params = *(cudaMemcpyFromSymbol_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.symbol);
                        crow.kind = (uint32_t)(params.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020:
                     {
                        auto &params = *(cudaMemcpyToArrayAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                     }
                     break;

                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020:
                     {
                        auto &params = *(cudaMemcpyFromArrayAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
                     {
                        auto &params = *(cudaMemcpy2DAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_v3020:
                     {
                        auto &params = *(cudaMemcpy2DToArrayAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(params.width);
                        crow.height = (uint32_t)(params.height);
                        crow.dst = fmt::format("{}", (void *)params.dst);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_v3020:
                     {
                        auto &params = *(cudaMemcpyFromArrayAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", (void *)params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                     }
                     break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020:
                    {
                        auto &params = *(cudaMemcpyToSymbolAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.symbol);
                        crow.src = fmt::format("{}", params.src);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                    }
                    break;
                case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020:
                    {
                        auto &params = *(cudaMemcpyFromSymbolAsync_v3020_params_st *)(cbInfo->functionParams);
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(params.count);
                        crow.dst = fmt::format("{}", params.dst);
                        crow.src = fmt::format("{}", params.symbol);
                        crow.kind = (uint32_t)(params.kind);
                        crow.stream = fmt::format("{}", (void *)params.stream);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                    }
                    break;
                default:
                    break;
            }
            logger.apiTable().insert(row);
        }
    }
    else if (domain = CUPTI_CB_DOMAIN_NVTX) {
        ApiTable::row row;
        row.pid = GetPid();
        row.tid = GetTid();
        row.start = clocktime_ns();
        row.end = row.start;
        static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
        row.apiName_id = markerId;
        row.args_id = EMPTY_STRING_ID;
        row.api_id = 0;

        CUpti_NvtxData* data = (CUpti_NvtxData*)cbInfo;

        switch (cbid) {
            case CUPTI_CBID_NVTX_nvtxMarkA:
                {
                    auto &params = *(nvtxMarkA_params_st *)(data->functionParams);
                    row.args_id = logger.stringTable().getOrCreate(params.message);
                    logger.apiTable().insertRoctx(row);
                }
                break;
            case CUPTI_CBID_NVTX_nvtxRangePushA:
                {
                    auto &params = *(nvtxRangePushA_params_st *)(data->functionParams);
                    row.args_id = logger.stringTable().getOrCreate(params.message);
                    logger.apiTable().pushRoctx(row);
                }
                break;
            case CUPTI_CBID_NVTX_nvtxRangePop:
                    logger.apiTable().popRoctx(row);
                break;
            default:
                break;
        }
    }
    std::call_once(register_once, atexit, Logger::rpdFinalize);
}


void CUPTIAPI CuptiDataSource::bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    *buffer = (uint8_t*)malloc(16 * 1024);
    *size = 16 * 1024;
    *maxNumRecords = 0;
}

void CUPTIAPI CuptiDataSource::bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    Logger &logger = Logger::singleton();
    int batchSize = 0;
    CUpti_Activity *it = NULL;

    // Cupti uses CLOCK_REALTIME for timestamps, which is not suitable for profiling
    //   Convert to a reliable timestamp domain, hopefully doing no additional damage
    timestamp_t t0, t1, t00;
    t0 = clocktime_ns();
    cuptiGetTimestamp(&t1);
    t00 = clocktime_ns();
    const timestamp_t toffset = (t0 >> 1) + (t00 >> 1) - t1;

    if (validSize > 0) {
        do {
            CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &it);
            if (status == CUPTI_SUCCESS) {
                OpTable::row row;

                switch(it->kind) {
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                        {
                            CUpti_ActivityMemcpy4 *record = (CUpti_ActivityMemcpy4 *) it;
                            row.gpuId = record->deviceId;
                            row.queueId = record->contextId;        // FIXME: this or stream
                            row.sequenceId = record->streamId;
                            //row.completionSignal = "";    //strcpy
                            strncpy(row.completionSignal, "", 18);
                            row.start = record->start + toffset;
                            row.end = record->end + toffset;
                            row.description_id = EMPTY_STRING_ID;
                            row.opType_id = logger.stringTable().getOrCreate("Memcpy");
                            row.api_id = record->correlationId;
                            logger.opTable().insert(row);
                        }
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMSET:
                        {
                            CUpti_ActivityMemset3 *record = (CUpti_ActivityMemset3 *) it;
                            row.gpuId = record->deviceId;
                            row.queueId = record->contextId;        // FIXME: this or stream
                            row.sequenceId = record->streamId;
                            //row.completionSignal = "";    //strcpy
                            strncpy(row.completionSignal, "", 18);
                            row.start = record->start + toffset;
                            row.end = record->end + toffset;
                            row.description_id = EMPTY_STRING_ID;
                            row.opType_id = logger.stringTable().getOrCreate("Memset");
                            row.api_id = record->correlationId;
                            logger.opTable().insert(row);
                        }
                        break;
                    case CUPTI_ACTIVITY_KIND_KERNEL:
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                        {
                            CUpti_ActivityKernel6 *record = (CUpti_ActivityKernel6 *) it;
                            const char *name = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "Kernel" : "ConcurrentKernel";
                            row.gpuId = record->deviceId;
                            row.queueId = record->contextId;	// FIXME: this or stream
                            row.sequenceId = record->streamId;
                            //row.completionSignal = "";    //strcpy
                            strncpy(row.completionSignal, "", 18);
                            row.start = record->start + toffset;
                            row.end = record->end + toffset;
                            row.description_id = logger.stringTable().getOrCreate(cxx_demangle(record->name));
                            row.opType_id = logger.stringTable().getOrCreate(name);
                            row.api_id = record->correlationId;
                            logger.opTable().insert(row);
                        }
                        break;
                    default:
                        break;

                }
                ++batchSize;
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                status;
            }
        } while (1);

      // report any records dropped from the queue
      size_t dropped;
      cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
      if (dropped != 0) {
          fprintf(stderr, "Dropped %u activity records\n", (unsigned int) dropped);
      }
    }
    free(buffer);
    std::call_once(registerAgain_once, atexit, Logger::rpdFinalize);
}


CudaApiIdList::CudaApiIdList()
{
    uint32_t cbid = 0;
    const char *name;
    while (cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, ++cbid, &name) == CUPTI_SUCCESS) {
        m_nameMap.emplace(name, cbid);
    }
}

uint32_t CudaApiIdList::mapName(const std::string &apiName)
{
    auto it = m_nameMap.find(apiName);
    if (it != m_nameMap.end()) {
        return it->second;
    }
    return 0;
}
