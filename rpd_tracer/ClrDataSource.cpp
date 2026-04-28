/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ClrDataSource.h"

#include <hip/hip_runtime.h>  // FIXME removeme
#include <hip/amd_detail/hip_profiler_ext.h>

#include <stdio.h>
#include <fmt/format.h>

#include "Logger.h"
#include "Utility.h"

namespace rpdtracer {

// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *ClrDataSourceFactory() { return new ClrDataSource(); }
}  // extern "C"


void ClrDataSource::init()
{
}

void ClrDataSource::end()
{
    flush();
}

void ClrDataSource::startTracing()
{
    fprintf(stderr, "ClrDataSource::startTracing()\n");
    // Work around teething bug
    hipDeviceProp_t devProp;
    (void)hipGetDeviceProperties(&devProp, 0);

    uint64_t startId;
    (void)hipProfilerEnableExt(&startId, 0);
    m_startIds.push_back(startId);
    fprintf(stderr, "ClrDataSource::startTracing(): %lu\n", startId);
}

void ClrDataSource::stopTracing()
{
    uint64_t endId;
    (void)hipProfilerDisableExt(&endId);
    m_endIds.push_back(endId);
    fprintf(stderr, "ClrDataSource::stopTracing(): %lu\n", endId);
}

void ClrDataSource::flush()
{
    fprintf(stderr, "ClrDataSource::flush()\n");
    const HipApiRecordExt* const* chunks;
    size_t chunk_count, chunk_size, total_count;
    (void)hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total_count);

    Logger &logger = Logger::singleton();
    static sqlite3_int64 domain_id = logger.stringTable().getOrCreate("hip");
    static uint64_t correlation_id = 0;

    for (size_t c = 0; c < chunk_count; ++c) {
        size_t n = (total_count - c * chunk_size < chunk_size)
                   ? total_count - c * chunk_size : chunk_size;
        for (size_t i = 0; i < n; ++i) {
            const HipApiRecordExt* r = &chunks[c][i];
            ApiTable::row row;
            row.pid = GetPid();
            row.tid = static_cast<int>(r->thread_id);
            row.start = static_cast<sqlite3_int64>(r->start_ns);
            row.end = static_cast<sqlite3_int64>(r->end_ns);
            row.domain_id = domain_id;
            row.category_id = EMPTY_STRING_ID;
            row.apiName_id = logger.stringTable().getOrCreate(r->api_name);
            row.args_id = EMPTY_STRING_ID;
            row.api_id = correlation_id;
            logger.apiTable().insert(row);


            ++correlation_id;
            if (r->has_gpu_activity) {
                static sqlite3_int64 dispatch_type_id = logger.stringTable().getOrCreate("KernelExecution");
                static sqlite3_int64 copy_type_id = logger.stringTable().getOrCreate("Memcpy");
                static sqlite3_int64 barrier_type_id = logger.stringTable().getOrCreate("Barrier");

                const HipGpuActivityExt* gpu = &r->gpu;
                while (gpu != nullptr) {
                    OpTable::row oprow;
                    oprow.gpuId = static_cast<int>(gpu->device_id);
                    oprow.queueId = static_cast<int>(gpu->queue_id);
                    oprow.sequenceId = 0;
                    oprow.start = static_cast<sqlite3_int64>(gpu->begin_ns);
                    oprow.end = static_cast<sqlite3_int64>(gpu->end_ns);
                    oprow.api_id = row.api_id;

                    if (gpu->op == HIP_OP_DISPATCH_EXT) {
                        oprow.opType_id = dispatch_type_id;
                        oprow.description_id = (gpu->kernel_name)
                            ? logger.stringTable().getOrCreate(gpu->kernel_name)
                            : EMPTY_STRING_ID;

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)r->stream);
                        krow.kernelName_id = oprow.description_id;
                        krow.gridX = static_cast<int>(gpu->grid_x);
                        krow.gridY = static_cast<int>(gpu->grid_y);
                        krow.gridZ = static_cast<int>(gpu->grid_z);
                        krow.workgroupX = static_cast<int>(gpu->block_x);
                        krow.workgroupY = static_cast<int>(gpu->block_y);
                        krow.workgroupZ = static_cast<int>(gpu->block_z);
                        logger.kernelApiTable().insert(krow);
                    } else if (gpu->op == HIP_OP_COPY_EXT) {
                        oprow.opType_id = copy_type_id;
                        static sqlite3_int64 sdma_id = logger.stringTable().getOrCreate("SDMA");
                        oprow.description_id = hipCopyKindIsSDMAExt((HipCopyKindExt)gpu->copy_kind)
                            ? sdma_id
                            : EMPTY_STRING_ID;

                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)r->stream);
                        crow.size = static_cast<int>(gpu->bytes);
                        crow.dst = fmt::format("{}", gpu->dst);
                        crow.src = fmt::format("{}", gpu->src);
                        crow.kind = static_cast<int>(gpu->copy_kind);
                        logger.copyApiTable().insert(crow);
                    } else {
                        oprow.opType_id = barrier_type_id;
                        oprow.description_id = EMPTY_STRING_ID;
                    }

                    logger.opTable().insert(oprow);
                    gpu = gpu->next;
                }
            }
        }
    }
}

}    // namespace rpdtracer
