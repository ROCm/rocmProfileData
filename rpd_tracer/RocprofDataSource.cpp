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
#include "RocprofDataSource.h"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>

#include "common/call_stack.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"

#include <vector>
#include <array>
#include <string>

#include <sqlite3.h>
#include <fmt/format.h>

#include "Logger.h"
#include "Utility.h"


// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *RocprofDataSourceFactory() { return new RocprofDataSource(); }
}  // extern "C"


//
// The plan:
//    Shared Class holds data common to all instances (should we ever need more than 1)
//    Anonymous namespace holds a ptr to the Shared Class.  Not member functio access needed
//    Class instances have a private object
//    Contexts have to be generated up-front
//        One context (always active) to observe code-object loading, etc
//        Class instances grab a context from an array.  For event callbacks and buffers
//

class RocprofDataSourceShared;
namespace
{
    RocprofDataSourceShared *s {nullptr};
    using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    using kernel_symbol_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
    using kernel_name_map_t = std::unordered_map<rocprofiler_kernel_id_t, const char *>;
    using rocprofiler::sdk::buffer_name_info;
    using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

    union ApiData {
        hipStream_t stream;
    };

    // extract copy args
            auto extract_copy_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                auto &crow = *(static_cast<CopyApiTable::row*>(cb_data));
                if (strcmp("dst", arg_name) == 0) {
                    crow.dst = std::string(arg_value_str);
                }
                else if (strcmp("src", arg_name) == 0) {
                    crow.src = std::string(arg_value_str);
                }
                else if (strcmp("sizeBytes", arg_name) == 0) {
                    crow.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
                }
                else if (strcmp("kind", arg_name) == 0) {
                    crow.kindStr = std::string(arg_value_str);
                    crow.kind = *(reinterpret_cast<const hipMemcpyKind*>(arg_value_addr));
                }
                else if (strcmp("stream", arg_name) == 0) {
                    crow.stream = std::string(arg_value_str);
                }
                return 0;
            };

    // extract kernel args
            auto extract_kernel_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                if (strcmp("stream", arg_name) == 0) {
                    auto &krow = *(static_cast<KernelApiTable::row*>(cb_data));
                    krow.stream = std::string(arg_value_str);
                }
                return 0;
            };

    // Extract stream args
            auto extract_stream_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                if (strcmp("stream", arg_name) == 0) {
                    auto &data = *(static_cast<ApiData*>(cb_data));
                    data.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
                }
                return 0;
            };


    // copy api calls
    bool isCopyApi(uint32_t id) {
        switch (id) {
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
                return true;
                break;
            default:
                ;
       }
       return false;
    }

    // kernel api calls
    bool isKernelApi(uint32_t id) {
        switch (id) {
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
                return true;
                break;
            default:
                ;
       }
       return false;
    }

    class RocprofApiIdList : public ApiIdList
    {
    public:
        RocprofApiIdList(buffer_name_info &names);
        uint32_t mapName(const std::string &apiName) override;
        std::vector<rocprofiler_tracing_operation_t> allEnabled();
    private:
        std::unordered_map<std::string, size_t> m_nameMap;
    };

} // namespace

class RocprofDataSourceShared
{
public:
    static RocprofDataSourceShared& singleton();

    rocprofiler_client_id_t *clientId {nullptr};
    rocprofiler_tool_configure_result_t cfg = rocprofiler_tool_configure_result_t{
                                            sizeof(rocprofiler_tool_configure_result_t),
                                            &RocprofDataSource::toolInit,
                                            &RocprofDataSource::toolFinialize,
                                            nullptr};

    // Contexts
    rocprofiler_context_id_t utilityContext = {0};
    std::array<rocprofiler_context_id_t,1> contexts = {0};
    std::array<RocprofDataSource*,1> instances = {nullptr};
    size_t nextContext = 0;	// first available context in contexts array

    // Buffers
    std::array<rocprofiler_buffer_id_t,1> client_buffers = {0};

    // Manage kernel names - #betterThanRoctracer

    kernel_symbol_map_t kernel_info = {};
    kernel_name_map_t kernel_names = {};

    // Manage buffer name - #betterThanRoctracer
    buffer_name_info name_info = {};

    // Agent info
    // <rocprofiler_profile_config_id_t.handle, rocprofiler_agent_v0_t>
    agent_info_map_t agents = {};

private:
    RocprofDataSourceShared() { s = this; }
    ~RocprofDataSourceShared() { s = nullptr; }
};

RocprofDataSourceShared &RocprofDataSourceShared::singleton()
{
    static RocprofDataSourceShared *instance = new RocprofDataSourceShared();	// Leak this
    return *instance;
}



class RocprofDataSourcePrivate
{
public:
    size_t id;
    //thread_local std::string stream;
    std::map<uint64_t, KernelApiTable::row> kernelrows;
    std::map<uint64_t, CopyApiTable::row> copyrows;

    // Circular buffer of api arguments - attach these when the buffers come in
    // avoid wraparound hopefully.  A sample heavily queued workload has about 6k in flight
    uint64_t apiDataSize { 1024 * 128 };	// 20x load factor - don't detect wrap, good luck

    std::vector<ApiData> apiData;
    std::mutex apiDataMutex;
    //std::atomic<uint64_t> apiDataHead{0}, apiDataTail{0};	// wrap detection

};


RocprofDataSource::RocprofDataSource()
: d(new RocprofDataSourcePrivate)
{
    RocprofDataSourceShared::singleton();	// CRITICAL: static init

    if (s->utilityContext.handle == 0) {
        // s->contexts have not been created.  Force registration
        auto ret = rocprofiler_force_configure(nullptr);
    }

    // assign ourselves then next available id and context
    assert(s->nextContext < s->contexts.size());
    d->id = s->nextContext++;
    s->instances[d->id] = this;
    d->apiData.reserve(d->apiDataSize);
}

RocprofDataSource::~RocprofDataSource()
{
    // FIXME: stop context?
    s->instances[d->id] = NULL;
    delete d;
}

void RocprofDataSource::init()
{
    stopTracing();
}

void RocprofDataSource::end() 
{
    flush();
}

void RocprofDataSource::startTracing()
{
    assert(s->contexts[d->id].handle != 0);
    rocprofiler_start_context(s->contexts[d->id]);
}

void RocprofDataSource::stopTracing()
{
    assert(s->contexts[d->id].handle != 0);
    rocprofiler_stop_context(s->contexts[d->id]);
}

void RocprofDataSource::flush()
{
    rocprofiler_flush_buffer(s->client_buffers[d->id]);
}


void RocprofDataSource::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    RocprofDataSource &instance = **(reinterpret_cast<RocprofDataSource**>(callback_data));

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            if (isCopyApi(record.operation) || isKernelApi(record.operation)) {
                // Capture the stream.  Will attach to the kernel and copy buffers when they arrive
                std::unique_lock<std::mutex> lock(instance.d->apiDataMutex);
                rocprofiler_iterate_callback_tracing_kind_operation_args(
                    record, extract_stream_args, 1/*max_deref*/
                    , &instance.d->apiData[record.correlation_id.internal % instance.d->apiDataSize]);
            }
        }
    }
}

#if 0
void RocprofDataSource::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    Logger &logger = Logger::singleton();
    RocprofDataSource &instance = **(reinterpret_cast<RocprofDataSource**>(callback_data));

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        thread_local sqlite3_int64 timestamp;	// FIXME: use userdata?  or stack?

        //fprintf(stderr, "%ld: HIP_RUNTIME_API %d %s %llu\n", record.correlation_id.internal, record.phase, std::string(s->name_info[record.kind][record.operation]).c_str(), clocktime_ns() - timestamp);

        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            timestamp = clocktime_ns();

            //---- Capture api args for copy and kernel ops
            if (isCopyApi(record.operation)) {
                rocprofiler_iterate_callback_tracing_kind_operation_args(
                    record, extract_copy_args, 1/*max_deref*/
                    , &instance.d->copyrows[record.correlation_id.internal]);
//fprintf(stderr, "====== copyrow for %ld\n", record.correlation_id.internal);
            }
            if (isKernelApi(record.operation)) {
                rocprofiler_iterate_callback_tracing_kind_operation_args(
                    record, extract_kernel_args, 1/*max_deref*/
                    , &instance.d->kernelrows[record.correlation_id.internal]);
            }
            //-----------------------------------------------
        }
        else {	     // ROCPROFILER_CALLBACK_PHASE_EXIT
            ApiTable::row row;

            //const char *name = fmt::format("{}::{}", record.kind, record.operation).c_str();
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(std::string(s->name_info[record.kind][record.operation]).c_str());
            row.pid = GetPid();
            row.tid = GetTid();
            row.start = timestamp;  // From TLS from preceding enter call
            row.end = clocktime_ns();
            row.apiName_id = name_id;
            row.args_id = EMPTY_STRING_ID;	// JSON up some args?
            row.api_id = record.correlation_id.internal;

#if 0
            auto info_data_cb = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {
                fprintf(stderr, "%d: %s (%s) -> %s\n", arg_num, arg_name, arg_type, arg_value_str);
                return 0;
            };

            rocprofiler_iterate_callback_tracing_kind_operation_args(
                    record, info_data_cb, 2/*max_deref*/, nullptr);
#endif
            logger.apiTable().insert(row);

            //---- Capture api args for copy and kernel ops
            if (isCopyApi(record.operation)) {
                // FIXME: do not remove here.  Used after the async operation
                // DO it anyway, wait for crash,  async SDMA should assert below
                instance.d->copyrows.erase(record.correlation_id.internal);
            }
            if (isKernelApi(record.operation)) {
                instance.d->kernelrows.erase(record.correlation_id.internal);
            }
            //-------------------------------------------------

        }
    } // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API
    else if (record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
        //fprintf(stderr, "KERNEL_DISPATCH %d (kind = %d  operation = %d)\n", record.phase, record.kind, record.operation);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            ;
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
            // enqueue callback - caller's thread
            auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
            auto &info = dispatch.dispatch_info;
            // Fetch data collected during api call

            std::string stream;

            if (instance.d->kernelrows.count(record.correlation_id.internal) > 0) {
                // This row can be missing.  Some copy api dispatch kernels under the hood
                auto &krow = instance.d->kernelrows.at(record.correlation_id.internal);
                stream = krow.stream;
            }
            else if (instance.d->copyrows.count(record.correlation_id.internal) > 0) {
                // Grab the stream from the copy row instead
                auto &crow = instance.d->copyrows.at(record.correlation_id.internal);
                stream = crow.stream;
            }
            KernelApiTable::row krow;
            krow.api_id = record.correlation_id.internal;	// FIXME, from nested hip call
            krow.stream = stream;
            krow.gridX = info.grid_size.x;
            krow.gridY = info.grid_size.y;
            krow.gridZ = info.grid_size.z;
            krow.workgroupX = info.workgroup_size.x;
            krow.workgroupY = info.workgroup_size.y;
            krow.workgroupZ = info.workgroup_size.z;
            krow.groupSegmentSize = info.group_segment_size;
            krow.privateSegmentSize = info.private_segment_size;
            krow.kernelName_id = logger.stringTable().getOrCreate(s->kernel_names.at(info.kernel_id));

            logger.kernelApiTable().insert(krow);
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
            // completion callback - runtime thread
            auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
            auto &info = dispatch.dispatch_info;
            static sqlite3_int64 name_id = logger.stringTable().getOrCreate("KernelExecution");

            OpTable::row row;
            row.gpuId = s->agents.at(info.agent_id.handle).logical_node_type_id;
            row.queueId = info.queue_id.handle;
            row.sequenceId = info.dispatch_id;
            strncpy(row.completionSignal, "", 18);
            row.start = dispatch.start_timestamp;
            row.end = dispatch.end_timestamp;
            row.description_id = logger.stringTable().getOrCreate(s->kernel_names.at(info.kernel_id));
            row.opType_id = name_id;
            row.api_id = record.correlation_id.internal;

            logger.opTable().insert(row);
        }
    }

    else if (record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
        //fprintf(stderr, "(%d::%d) MEMORY_COPY %d (kind = %d  operation = %d)\n", GetPid(), GetTid(), record.phase, record.kind, record.operation);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
            auto &copy = *(static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(record.payload));

            // Fetch data collected during api call
            // FIXME async?  May need to remove it here rather than above
//fprintf(stderr, "++++ looking for %ld\n", record.correlation_id.internal);
            auto &crow = instance.d->copyrows.at(record.correlation_id.internal);
            //CopyApiTable::row crow;
 
            crow.api_id = record.correlation_id.internal; // FIXME, from nested hip call. matches?
            // FIXME: split copies.  Crow has total size.  This record has a segment size
            //crow.size = (uint32_t)(copy.bytes);
            //crow.dst = ;
            //crow.src = ;
            // Use node_id.  Will not match node_type_id from ops.  Can express cpu location
            crow.dstDevice = s->agents.at(copy.dst_agent_id.handle).logical_node_id;
            crow.srcDevice = s->agents.at(copy.src_agent_id.handle).logical_node_id;
            //crow.kind = ;

            logger.copyApiTable().insert(crow);

            static sqlite3_int64 name_id = logger.stringTable().getOrCreate("Memcpy");
            OpTable::row row;
            //row.gpuId = mapDeviceId(record->device_id);
            row.gpuId = 0;	// FIXME intercept hsa to figure out node?
            row.queueId = 0;
            row.sequenceId = 0;
            strncpy(row.completionSignal, "", 18);
            row.start = copy.start_timestamp;
            row.end = copy.end_timestamp;
            row.description_id = logger.stringTable().getOrCreate(crow.kindStr);
            row.opType_id = name_id;
            row.api_id = record.correlation_id.internal;
            logger.opTable().insert(row);

            // dispose the copyapi row
            //instance.d->copyrows.erase(record.correlation_id.internal);
            // FIXME can not dispose after use.  Copyapi -> copyop can be 1:n
        }
    }
}
#endif

void RocprofDataSource::roctx_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        return;

    Logger &logger = Logger::singleton();

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API) {
        ApiTable::row row;
        row.pid = GetPid();
        row.tid = GetTid();
        row.start = clocktime_ns();
        row.end = row.start;
        static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
        row.apiName_id = markerId;
        row.args_id = EMPTY_STRING_ID;
        row.api_id = 0;

        auto &data = *(static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload));

        switch (record.operation) {
            case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
                row.args_id = logger.stringTable().getOrCreate(data.args.roctxMarkA.message);
                logger.apiTable().insertRoctx(row);
            break;
            case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
                row.args_id = logger.stringTable().getOrCreate(data.args.roctxRangePushA.message);
                logger.apiTable().pushRoctx(row);
            break;
            case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop:
                logger.apiTable().popRoctx(row);
            break;
        };
    }
}


#if 1
void RocprofDataSource::buffer_callback(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id, rocprofiler_record_header_t** headers, size_t num_headers, void* user_data, uint64_t drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");
    RocprofDataSource &instance = **(reinterpret_cast<RocprofDataSource**>(user_data));

    Logger &logger = Logger::singleton();

    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];

        if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING) {
            if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {

                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
                auto& dispatch = record->dispatch_info;
                // FIXME: op name hack
                static sqlite3_int64 name_id = logger.stringTable().getOrCreate("KernelExecution");
                sqlite3_int64 desc_id = logger.stringTable().getOrCreate(s->kernel_names.at(record->dispatch_info.kernel_id));

                OpTable::row row; 
                row.gpuId = s->agents.at(dispatch.agent_id.handle).logical_node_type_id;
                row.queueId = dispatch.queue_id.handle;
                row.sequenceId = 0;
                strncpy(row.completionSignal, "", 18);
                row.start = record->start_timestamp;
                row.end = record->end_timestamp;
                row.description_id = desc_id;
                row.opType_id = name_id;
                row.api_id = record->correlation_id.internal;

                logger.opTable().insert(row);

                // piece together a kernelapi entry
                KernelApiTable::row krow;
                krow.api_id = record->correlation_id.internal;
                {
                    std::unique_lock<std::mutex> lock(instance.d->apiDataMutex);
                    krow.stream = fmt::format("{}", (void *)instance.d->apiData[record->correlation_id.internal % instance.d->apiDataSize].stream);
                }
                krow.gridX = dispatch.grid_size.x;
                krow.gridY = dispatch.grid_size.y;
                krow.gridZ = dispatch.grid_size.z;
                krow.workgroupX = dispatch.workgroup_size.x;
                krow.workgroupY = dispatch.workgroup_size.y;
                krow.workgroupZ = dispatch.workgroup_size.z;
                krow.groupSegmentSize = dispatch.group_segment_size;
                krow.privateSegmentSize = dispatch.private_segment_size;
                krow.kernelName_id = desc_id;

                logger.kernelApiTable().insert(krow);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {

                auto &copy = *(static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload));
                sqlite3_int64 name_id = logger.stringTable().getOrCreate(std::string(s->name_info[copy.kind][copy.operation]).c_str());
                sqlite3_int64 desc_id = logger.stringTable().getOrCreate("");

                // Add the op entry
                OpTable::row row;
                row.gpuId = 0;
                row.queueId = 0;	// FIXME, all wrong
                row.sequenceId = 0;
                strncpy(row.completionSignal, "", 18);
                row.start = copy.start_timestamp;
                row.end = copy.end_timestamp;
                row.description_id = desc_id;
                row.opType_id = name_id;
                row.api_id = copy.correlation_id.internal;

                logger.opTable().insert(row);

                // piece together a copyapi entry
                CopyApiTable::row crow;
                crow.api_id = copy.correlation_id.internal;
                crow.size = (uint32_t)(copy.bytes);
                {
                    std::unique_lock<std::mutex> lock(instance.d->apiDataMutex);
                    crow.stream = fmt::format("{}", (void *)instance.d->apiData[copy.correlation_id.internal % instance.d->apiDataSize].stream);
                }
                //crow.stream = s->stream;
                // Use node_id.  Will not match node_type_id from ops.  Can express cpu location
                crow.dstDevice = s->agents.at(copy.dst_agent_id.handle).logical_node_id;
                crow.srcDevice = s->agents.at(copy.src_agent_id.handle).logical_node_id;
                crow.kind = name_id;
                crow.sync = true;

                logger.copyApiTable().insert(crow);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) {
                auto &hipapi = *(static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload));

                // Add an api table entry
                sqlite3_int64 name_id = logger.stringTable().getOrCreate(std::string(s->name_info[hipapi.kind][hipapi.operation]).c_str());

                ApiTable::row row;
                row.pid = GetPid();
                row.tid = hipapi.thread_id;
                row.start = hipapi.start_timestamp;
                row.end = hipapi.end_timestamp;
                row.apiName_id = name_id;
                row.args_id = EMPTY_STRING_ID;
                row.api_id = hipapi.correlation_id.internal;

                logger.apiTable().insert(row);
            }
        }
    }
}
#endif

void RocprofDataSource::code_object_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    //fprintf(stderr, "code_object_callback\n");
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
// FIXME
            //auto flush_status = rocprofiler_flush_buffer(client_buffer);
            //if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            //    ;
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            s->kernel_info.emplace(data->kernel_id, *data);
            s->kernel_names.emplace(data->kernel_id, cxx_demangle(data->kernel_name));
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // FIXME: clear these?  At minimum need kernel names at shutdown, async completion
            //s->kernel_info.erase(data->kernel_id);
            //s->kernel_names.erase(data->kernel_id);
        }
    }
}


std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
{
    std::vector<rocprofiler_agent_v0_t> agents;

    // Callback used by rocprofiler_query_available_agents to return
    // agents on the device. This can include CPU agents as well. We
    // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            //if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
            agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Query the agents, only a single callback is made that contains a vector
    // of all agents.
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents)));
    return agents;
}


//
//
// Static setup
//
//


extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    RocprofDataSourceShared::singleton();	// CRITICAL: static init

    id->name = "rpd_tracer";
    s->clientId = id;

    // return pointer to configure data
    return &s->cfg;
}


int RocprofDataSource::toolInit(rocprofiler_client_finalize_t finialize_func, void* tool_data)
{
    //s->name_info = common::get_buffer_tracing_names();
    s->name_info = rocprofiler::sdk::get_buffer_tracing_names();  // FIXME: decide
    //s->name_info = rocprofiler::sdk::get_callback_tracing_names();

    auto agent_info = get_gpu_device_agents();

    for (auto agent : agent_info) {
        s->agents[agent.id.handle] = agent;
    }

    // Common context
    //-------------------------------------------------------
    rocprofiler_create_context(&s->utilityContext);

    // Code Objects
    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    rocprofiler_configure_callback_tracing_service(s->utilityContext,
                                                   ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                   code_object_ops.data(),
                                                   code_object_ops.size(),
                                                   RocprofDataSource::code_object_callback,
                                                   nullptr);

    {
        int isValid = 0;
        rocprofiler_context_is_valid(s->utilityContext, &isValid);
        if (isValid == 0) {
            s->utilityContext.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
    }

    rocprofiler_start_context(s->utilityContext);

    // select some api calls to omit, in the most inconvenient way possible
    // #betterThanRoctracer

    RocprofApiIdList apiList(s->name_info);
    apiList.setInvertMode(true);  // Omit the specified api
    apiList.add("hipGetDevice");
    apiList.add("hipSetDevice");
    apiList.add("hipGetLastError");
    apiList.add("__hipPushCallConfiguration");
    apiList.add("__hipPopCallConfiguration");
    apiList.add("hipCtxSetCurrent");
    apiList.add("hipEventRecord");
    apiList.add("hipEventQuery");
    apiList.add("hipGetDeviceProperties");
    apiList.add("hipPeekAtLastError");
    apiList.add("hipModuleGetFunction");
    apiList.add("hipEventCreateWithFlags");

    // Get a vector of the enabled api calls
    auto apis = apiList.allEnabled();

    // Instance s->contexts
    //-------------------------------------------------------

    //for (auto &context : s->contexts) {
    for (int i = 0; i < s->contexts.size(); ++i) {
        auto &context = s->contexts[i];
        auto &buffer = s->client_buffers[i];
        auto instance = &s->instances[i];

        rocprofiler_create_context(&context);

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                   apis.data(),
                                                   apis.size(),
                                                   api_callback,
                                                   instance);

#if 0
        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   instance);

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   instance);
#endif

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                   nullptr,
                                                   0,
                                                   roctx_callback,
                                                   instance);
#if 1
        // Buffers
        constexpr auto buffer_size_bytes      = 0x40000;
        constexpr auto buffer_watermark_bytes = buffer_size_bytes / 2;

        rocprofiler_create_buffer(context,
                                  buffer_size_bytes,
                                  buffer_watermark_bytes,
                                  ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                  RocprofDataSource::buffer_callback,
                                  //nullptr, /*tool_data,*/
                                  instance,
                                  &buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                     nullptr,
                                                     0,
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                     nullptr,
                                                     0,
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                     apis.data(),
                                                     apis.size(),
                                                     buffer);

        auto client_thread = rocprofiler_callback_thread_t{};
        rocprofiler_create_callback_thread(&client_thread);
        rocprofiler_assign_callback_thread(buffer, client_thread);
#endif

        int isValid = 0;
        rocprofiler_context_is_valid(context, &isValid);
        if (isValid == 0) {
            context.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
        rocprofiler_start_context(context);
    }

    return 0;
}

void RocprofDataSource::toolFinialize(void* tool_data)
{
    // This seems to happen pretty early.  So simulate a shutdown and disable context
    //fprintf(stderr, "RocprofDataSource::toolFinalize\n");

    //end();	// FIXME: singleton - figure out startup order and teardown order

    // FIXME: kernel code objects are already being removed by this point
    //        keeping names (only) around to remedy this

    rocprofiler_stop_context(s->utilityContext);
    s->utilityContext.handle = 0;    // save us from ourselves
    for (auto &context : s->contexts) {
        rocprofiler_stop_context(context);
        context.handle = 0;
    }
}


RocprofApiIdList::RocprofApiIdList(buffer_name_info &names)
: m_nameMap()
{
    auto &hipapis = names[ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API].operations;

    for (size_t i = 0; i < hipapis.size(); ++i) {
        m_nameMap.emplace(hipapis[i], i);
    }
}

uint32_t RocprofApiIdList::mapName(const std::string &apiName)
{
    auto it = m_nameMap.find(apiName);
    if (it != m_nameMap.end()) {
        return it->second;
    }
    return 0;
}

std::vector<rocprofiler_tracing_operation_t> RocprofApiIdList::allEnabled()
{
    std::vector<rocprofiler_tracing_operation_t> oplist;
    for (auto &it : m_nameMap) {
        if (contains(it.second))
            oplist.push_back(it.second);
    }
    return oplist;
}
