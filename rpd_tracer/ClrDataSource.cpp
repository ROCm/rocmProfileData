/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ClrDataSource.h"

#include <hip/amd_detail/hip_profiler_ext.h>

#include <stdio.h>

#include "Logger.h"

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
}

void ClrDataSource::startTracing()
{
    uint64_t startId;
    hipProfilerEnableExt(&startId);
    m_startIds.push_back(startId);
}

void ClrDataSource::stopTracing()
{
    uint64_t endId;
    hipProfilerDisableExt(&endId);
    m_endIds.push_back(endId);
}

void ClrDataSource::flush()
{
return;
    const HipApiRecordExt* const* chunks;
    size_t chunk_count, chunk_size, total_count;
    hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total_count);

    for (size_t c = 0; c < chunk_count; ++c) {
        size_t n = (total_count - c * chunk_size < chunk_size)
                   ? total_count - c * chunk_size : chunk_size;
        for (size_t i = 0; i < n; ++i) {
            const HipApiRecordExt* r = &chunks[c][i];
            fprintf(stderr, "api=%s tid=%lu start=%lu end=%lu",
                    r->api_name, r->thread_id, r->start_ns, r->end_ns);
            if (r->has_gpu_activity) {
                fprintf(stderr, " gpu_begin=%lu gpu_end=%lu gpu_op=%u",
                        r->gpu.begin_ns, r->gpu.end_ns, (unsigned)r->gpu.op);
                if (r->gpu.op == HIP_OP_DISPATCH_EXT && r->gpu.kernel_name)
                    fprintf(stderr, " kernel=%s", r->gpu.kernel_name);
                else if (r->gpu.op == HIP_OP_COPY_EXT)
                    fprintf(stderr, " bytes=%lu", r->gpu.bytes);
            }
            fprintf(stderr, "\n");
        }
    }
}

}    // namespace rpdtracer
