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

#include "common/call_stack.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/name_info.hpp"

#include <vector>
#include <array>

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
    using common::buffer_name_info;
    using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;
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
    size_t nextContext = 0;	// first available context in contexts array

    // Buffers
    //rocprofiler_buffer_id_t client_buffer = {};

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
};


RocprofDataSource::RocprofDataSource()
: d(new RocprofDataSourcePrivate)
{
    RocprofDataSourceShared::singleton();	// CRITICAL: static init

    if (s->utilityContext.handle == 0) {
        // s->contexts have not been created.  Force registration
        auto ret = rocprofiler_force_configure(nullptr);
        //fprintf(stderr, "******  rocprofiler_force_configure: %d\n", ret);
    }

    // assign ourselves then next available id and context
    assert(s->nextContext < s->contexts.size());
    d->id = s->nextContext++;
}

RocprofDataSource::~RocprofDataSource()
{
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
    // no buffering - no flushing
}

void RocprofDataSource::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    Logger &logger = Logger::singleton();

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        thread_local sqlite3_int64 timestamp;	// FIXME: use userdata?  or stack?

        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
                timestamp = clocktime_ns();
            //fprintf(stderr, "HIP_RUNTIME_API %d %s\n", record.phase, std::string(s->name_info[record.kind][record.operation]).c_str());
        }
        else {	     // ROCPROFILER_CALLBACK_PHASE_EXIT
            //fprintf(stderr, "HIP_RUNTIME_API %d %s %llu\n", record.phase, std::string(s->name_info[record.kind][record.operation]).c_str(), clocktime_ns() - timestamp);
            char buff[4096];
            ApiTable::row row;

            //const char *name = fmt::format("{}::{}", record.kind, record.operation).c_str();
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(std::string(s->name_info[record.kind][record.operation]).c_str());
            row.pid = GetPid();
            row.tid = GetTid();
            row.start = timestamp;  // From TLS from preceding enter call
            row.end = clocktime_ns();
            row.apiName_id = name_id;
            row.args_id = EMPTY_STRING_ID;
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
        }
    } // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API
    else if (record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
        //fprintf(stderr, "KERNEL_DISPATCH %d (kind = %d  operation = %d)\n", record.phase, record.kind, record.operation);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
            // enqueue callback - caller's thread
            auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
            auto &info = dispatch.dispatch_info;

//fprintf(stderr, "%s\n", std::string(s->name_info[record.kind][record.operation]).c_str());
//fprintf(stderr, "%d::%d\n", record.kind, record.operation);

            KernelApiTable::row krow;
            krow.api_id = record.correlation_id.internal;	// FIXME, from nested hip call
            krow.stream = fmt::format("FIXME");
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
            //row.gpuId = mapDeviceId(record->device_id);
            row.gpuId = s->agents.at(info.agent_id.handle).logical_node_type_id;
            row.queueId = info.queue_id.handle;
            row.sequenceId = 0;
            strncpy(row.completionSignal, "", 18);
            //row.start = record->begin_ns + toffset;	FIXME
            //row.end = record->end_ns + toffset;
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
            CopyApiTable::row crow;
            crow.api_id = record.correlation_id.internal;       // FIXME, from nested hip call
            crow.size = (uint32_t)(copy.bytes);
            // Use node_id.  Will not match node_type_id from ops.  Can express cpu location
            crow.dstDevice = s->agents.at(copy.dst_agent_id.handle).logical_node_id;
            crow.srcDevice = s->agents.at(copy.src_agent_id.handle).logical_node_id;
            crow.kind = EMPTY_STRING_ID;
            crow.sync = true;

            logger.copyApiTable().insert(crow);

            static sqlite3_int64 name_id = logger.stringTable().getOrCreate("Memcpy");
            OpTable::row row;
            //row.gpuId = mapDeviceId(record->device_id);
            row.gpuId = 0;	// FIXME
            row.queueId = 0;
            row.sequenceId = 0;
            strncpy(row.completionSignal, "", 18);
            //row.start = record->begin_ns + toffset;   FIXME
            //row.end = record->end_ns + toffset;
            row.start = copy.start_timestamp;
            row.end = copy.end_timestamp;
            row.description_id = EMPTY_STRING_ID;
            row.opType_id = name_id;
            row.api_id = record.correlation_id.internal;

            logger.opTable().insert(row);

            //fprintf(stderr, "(%d::%d) copy %ld (%ld -> %ld)\n", GetPid(), GetTid(), copy.end_timestamp - copy.start_timestamp, copy.start_timestamp, copy.end_timestamp);
        }
    }
}


#if 0
void RocprofDataSource::buffer_callback(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id, rocprofiler_record_header_t** headers, size_t num_headers, void* user_data, uint64_t drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    fprintf(stderr, "buffer_callback - %ld headers\n", num_headers);

    Logger &logger = Logger::singleton();

    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];

        if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING) {
            if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {

                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
                // FIXME: op name hack
                static sqlite3_int64 name_id = logger.stringTable().getOrCreate("KernelExecution");
                sqlite3_int64 desc_id = logger.stringTable().getOrCreate(s->kernel_names.at(record->dispatch_info.kernel_id));

                OpTable::row row; 
                row.gpuId = record->dispatch_info.agent_id.handle;
                row.queueId = record->dispatch_info.queue_id.handle;
                row.sequenceId = 0;
                strncpy(row.completionSignal, "", 18);
                row.start = record->start_timestamp;
                row.end = record->end_timestamp;
                row.description_id = desc_id;
                row.opType_id = name_id;
                row.api_id = record->correlation_id.internal;

                logger.opTable().insert(row);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {

                auto *record = static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);
                sqlite3_int64 name_id = logger.stringTable().getOrCreate(fmt::format("{}::{}", record->kind, record->operation).c_str());
                sqlite3_int64 desc_id = logger.stringTable().getOrCreate("");

                OpTable::row row;
                row.gpuId = record->src_agent_id.handle;
                row.queueId = record->dst_agent_id.handle;	// FIXME, all wrong
                row.sequenceId = 0;
                strncpy(row.completionSignal, "", 18);
                row.start = record->start_timestamp;
                row.end = record->end_timestamp;
                row.description_id = desc_id;
                row.opType_id = name_id;
                row.api_id = record->correlation_id.internal;

                logger.opTable().insert(row);
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
    s->name_info = common::get_buffer_tracing_names();

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



    // Instance s->contexts
    //-------------------------------------------------------

    for (auto &context : s->contexts) {

        rocprofiler_create_context(&context);

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   nullptr);
                                                   //tool_data);

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   nullptr);
                                                   //tool_data);

        rocprofiler_configure_callback_tracing_service(context,
                                                   ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   nullptr);
                                                   //tool_data);

        int isValid = 0;
        rocprofiler_context_is_valid(context, &isValid);
        if (isValid == 0) {
            context.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
        rocprofiler_start_context(context);
    }

#if 0
    constexpr auto buffer_size_bytes      = 0x40000;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes / 2;

    rocprofiler_create_buffer(client_ctx,
                              buffer_size_bytes,
                              buffer_watermark_bytes,
                              ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                              RocprofDataSource::buffer_callback,
                              nullptr, /*tool_data,*/
                              &client_buffer);

    rocprofiler_configure_buffer_tracing_service(client_ctx,
                                                 ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                 nullptr,
                                                 0,
                                                 client_buffer);

    rocprofiler_configure_buffer_tracing_service(client_ctx,
                                                 ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                 nullptr,
                                                 0,
                                                 client_buffer);

    auto client_thread = rocprofiler_callback_thread_t{};
    rocprofiler_create_callback_thread(&client_thread);
    rocprofiler_assign_callback_thread(client_buffer, client_thread);
#endif

    return 0;
}

void RocprofDataSource::toolFinialize(void* tool_data)
{
    // This seems to happen pretty early.  So simulate a shutdown and disable context
    fprintf(stderr, "RocprofDataSource::toolFinalize\n");

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
