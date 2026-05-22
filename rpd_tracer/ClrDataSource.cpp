/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ClrDataSource.h"

#include <hip/hip_runtime.h>
#include <fmt/format.h>

#include "Logger.h"
#include "UStringCache.h"
#include "Utility.h"

namespace rpdtracer {

// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *ClrDataSourceFactory() { return new ClrDataSource(); }
}  // extern "C"

// FIXME: can we avoid shutdown corruption?
// Other rocm libraries crashing on unload
// libsqlite unloading before we are done using it

void ClrDataSource::init()
{
    m_apiList.setInvertMode(true);  // Omit the specified api
    m_apiList.add("hipGetDevice");
    m_apiList.add("hipSetDevice");
    m_apiList.add("hipGetLastError");
    m_apiList.add("__hipPushCallConfiguration");
    m_apiList.add("__hipPopCallConfiguration");
    m_apiList.add("hipCtxSetCurrent");
    m_apiList.add("hipGetDevicePropertiesR0600");
    m_apiList.add("hipGetDeviceCount");
    m_apiList.add("hipDeviceGetAttribute");
    m_apiList.add("hipRuntimeGetVersion");
    m_apiList.add("hipPeekAtLastError");
    m_apiList.add("hipModuleGetFunction");
    m_apiList.loadUserPrefs();

    // Must be registered before hipProfilerEnableExt
    (void)hipProfilerRegisterChunkCallbackExt(chunkCallback, this);
}

void ClrDataSource::end()
{
    flush();
}

void ClrDataSource::startTracing()
{
    // Work around teething bug
    hipDeviceProp_t devProp;
    (void)hipGetDeviceProperties(&devProp, 0);

    uint64_t startId;
    (void)hipProfilerEnableExt(&startId, 0);
    m_ranges.push_back({startId, 0});
}

void ClrDataSource::stopTracing()
{
    uint64_t endId;
    (void)hipProfilerDisableExt(&endId);
    m_ranges.back().end = endId;
}


void ClrDataSource::chunkCallback(const hipApiRecordExt* records, uint32_t count,
                                  uint32_t chunk_id, void* user_data)
{
    const timestamp_t begin_time = clocktime_ns();
    ClrDataSource* self = static_cast<ClrDataSource*>(user_data);
    for (uint32_t i = 0; i < count; ++i)
        self->processRecord(&records[i]);
    // chunk_id is the slab index, not records[0].chunk_id — use the last
    // record's chunk_id so flush() correctly skips all freed slabs.
    self->m_processedCount = records[count - 1].chunk_id + 1;
    const timestamp_t end_time = clocktime_ns();
    char buff[128];
    std::snprintf(buff, sizeof(buff), "count=%u | total=%zu", count, self->m_processedCount);
    Logger::singleton().createOverheadRecord(begin_time, end_time, "ClrDataSource::chunkCallback", buff);
}


static std::string formatKernelArgs(const uint8_t* args, uint32_t size)
{
    if (!args || size == 0)
        return "";
    const uint32_t MAX_BYTES = 64;
    std::string hex = "0x";
    uint32_t n = (size < MAX_BYTES) ? size : MAX_BYTES;
    for (uint32_t i = 0; i < n; ++i)
        hex += fmt::format("{:02x}", args[i]);
    if (size > MAX_BYTES)
        hex += "...";
    return fmt::format("{{\"args\":\"{}\"}}", hex);
}


void ClrDataSource::processRecord(const hipApiRecordExt* r)
{
    bool in_range = false;
    for (const auto& range : m_ranges) {
        uint64_t id = r->chunk_id;
        if (id >= range.start && (range.end == 0 || id < range.end)) {
            in_range = true;
            break;
        }
    }
    if (!in_range)
        return;

    if (!m_apiList.contains(r->api_name))
        return;

    cacheIds();
    Logger &logger = Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = static_cast<int>(static_cast<sqlite3_int64>(r->thread_id)); // FIXME need OS tid
    row.start = static_cast<sqlite3_int64>(r->start_ns);
    row.end = static_cast<sqlite3_int64>(r->end_ns);
    row.domain_id = m_domainId;
    row.category_id = EMPTY_STRING_ID;
    row.apiName_id = logger.stringTable().getOrCreate(r->api_name);
    row.args_id = EMPTY_STRING_ID;
    row.api_id = m_correlationId.fetch_add(1);

    {
        static thread_local rpdtracer::UStringCache t_ustringCache;
        uint64_t gen = logger.storageGeneration();
        if (r->has_gpu_activity && r->gpu.op == HIP_OP_DISPATCH_EXT) {
            std::string args = formatKernelArgs(r->gpu.kernel_args, r->gpu.kernel_args_size);
            if (!args.empty())
                row.args_id = t_ustringCache.lookup(args, logger.ustringTable(), gen);
        } else if (!r->has_gpu_activity && r->size > 0) {
            row.args_id = t_ustringCache.lookup(
                fmt::format("{{\"size\":{},\"dst\":\"{}\",\"src\":\"{}\"}}", r->size, r->memory1, r->memory2),
                logger.ustringTable(), gen);
        }
    }

    logger.apiTable().insert(row);

    if (r->has_gpu_activity) {
        const hipGpuActivityExt* gpu = &r->gpu;
        while (gpu != nullptr) {
            OpTable::row oprow;
            oprow.gpuId = static_cast<int>(gpu->device_id);
            oprow.queueId = static_cast<int>(gpu->queue_id);
            oprow.sequenceId = 0;
            oprow.start = static_cast<sqlite3_int64>(gpu->begin_ns);
            oprow.end = static_cast<sqlite3_int64>(gpu->end_ns);
            oprow.api_id = row.api_id;

            if (gpu->op == HIP_OP_DISPATCH_EXT) {
                oprow.opType_id = m_dispatchTypeId;
                oprow.description_id = (gpu->kernel_name && !gpu->is_graph)
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
                oprow.opType_id = m_copyTypeId;
                if (hipCopyKindIsSDMAExt((HipCopyKindExt)gpu->copy_kind))
                    oprow.description_id = m_sdmaId;
                else if (gpu->copy_kind == HIP_COPY_KIND_FILL_EXT)
                    oprow.description_id = m_fillId;
                else
                    oprow.description_id = EMPTY_STRING_ID;

                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)r->stream);
                crow.size = static_cast<int>(gpu->bytes);
                crow.dst = fmt::format("{}", gpu->dst);
                crow.src = fmt::format("{}", gpu->src);
                crow.kind = static_cast<int>(gpu->copy_kind);
                logger.copyApiTable().insert(crow);
            } else {
                oprow.opType_id = m_barrierTypeId;
                oprow.description_id = EMPTY_STRING_ID;
            }

            logger.opTable().insert(oprow);
            gpu = gpu->next;
        }
    } else if (r->size > 0) {
        CopyApiTable::row crow;
        crow.api_id = row.api_id;
        crow.stream = fmt::format("{}", (void*)r->stream);
        crow.size = static_cast<int>(r->size);
        crow.dst = fmt::format("{}", r->memory1);
        crow.src = fmt::format("{}", r->memory2);
        crow.kind = 0;
        logger.copyApiTable().insert(crow);
    }
}


void ClrDataSource::cacheIds()
{
    if (m_idsCached)
        return;
    Logger &logger = Logger::singleton();
    m_domainId = logger.stringTable().getOrCreate("hip");
    m_dispatchTypeId = logger.stringTable().getOrCreate("KernelExecution");
    m_copyTypeId = logger.stringTable().getOrCreate("Memcpy");
    m_barrierTypeId = logger.stringTable().getOrCreate("Barrier");
    m_sdmaId = logger.stringTable().getOrCreate("SDMA");
    m_fillId = logger.stringTable().getOrCreate("Fill");
    m_idsCached = true;
}

void ClrDataSource::reset()
{
    m_idsCached = false;
}

void ClrDataSource::flush()
{
    const timestamp_t cb_begin_time = clocktime_ns();

    const hipApiRecordExt* const* chunks;
    size_t chunk_count, chunk_size, total_count;
    (void)hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total_count);

    size_t start_chunk = (chunk_size > 0) ? m_processedCount / chunk_size : 0;
    size_t start_index = (chunk_size > 0) ? m_processedCount % chunk_size : 0;

    for (size_t c = start_chunk; c < chunk_count; ++c) {
        size_t n = (total_count - c * chunk_size < chunk_size)
                   ? total_count - c * chunk_size : chunk_size;
        for (size_t i = (c == start_chunk ? start_index : 0); i < n; ++i)
            processRecord(&chunks[c][i]);
    }

    size_t incremental_count = total_count - m_processedCount;
    m_processedCount = total_count;

    const timestamp_t cb_end_time = clocktime_ns();
    Logger &logger = Logger::singleton();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%lu | total=%lu", incremental_count, total_count);
    logger.createOverheadRecord(cb_begin_time, cb_end_time, "ClrDataSource::flush", buff);
}

}    // namespace rpdtracer
