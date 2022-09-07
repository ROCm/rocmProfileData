/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "CuptiDataSource.h"

#include <sqlite3.h>
#include <fmt/format.h>

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "generated_nvtx_meta.h"

#include "Logger.h"
#include "Utility.h"


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
    // FIXME: cuptiSubscribe may fail with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
    cuptiSubscribe(&m_subscriber, (CUpti_CallbackFunc)api_callback, nullptr);
    cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_NVTX);

    // Async Activity
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD);

    cuptiActivityRegisterCallbacks(CuptiDataSource::bufferRequested, CuptiDataSource::bufferCompleted);
    //cuptiActivityRegisterTimestampCallback();	// Cuda 11.6 :(
}

void CuptiDataSource::end()
{
    cuptiActivityFlushAll(1);
}

void CuptiDataSource::startTracing()
{
    //printf("# START ############################# %d\n", GetTid());
}

void CuptiDataSource::stopTracing()
{
    //printf("# STOP #############################\n");
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
                        krow.kernelName_id = logger.stringTable().getOrCreate(cxx_demangle(cbInfo->symbolName));

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
                //case HIP:
                //    break;
                //case HIP:
                //    break;
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
        row.apiName_id = logger.stringTable().getOrCreate(std::string("UserMarker"));   // FIXME: can cache
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
#if 0
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
#endif
}

void CUPTIAPI CuptiDataSource::bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    Logger &logger = Logger::singleton();
    int batchSize = 0;
    CUpti_Activity *it = NULL;

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
                            row.start = record->start;
                            row.end = record->end;
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
                            row.start = record->start;
                            row.end = record->end;
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
                            row.start = record->start;
                            row.end = record->end;
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
          printf("Dropped %u activity records\n", (unsigned int) dropped);
      }
    }
    free(buffer);
    std::call_once(registerAgain_once, atexit, Logger::rpdFinalize);
}

