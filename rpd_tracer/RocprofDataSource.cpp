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

#include <sqlite3.h>
#include <fmt/format.h>

#include "Logger.h"
#include "Utility.h"


// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *RocprofDataSourceFactory() { return new RocprofDataSource(); }
}  // extern "C"


namespace
{
    rocprofiler_client_id_t *clientId {nullptr};
    auto cfg = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &RocprofDataSource::toolInit,
                                            &RocprofDataSource::toolFinialize,
                                            nullptr};
    rocprofiler_context_id_t client_ctx = {0};
    rocprofiler_buffer_id_t client_buffer = {};

    // Manage kernel names - #betterThanRoctracer
    using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    using kernel_symbol_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
    using kernel_name_map_t = std::unordered_map<rocprofiler_kernel_id_t, const char *>;
    kernel_symbol_map_t kernel_info = {};
    kernel_name_map_t kernel_names = {};

    // Manage buffer name - #betterThanRoctracer
    using common::buffer_name_info;
    buffer_name_info name_info = {};

} // namespace





void RocprofDataSource::init()
{
    fprintf(stderr, "RocprofDataSource::init\n");
    stopTracing();
}

void RocprofDataSource::end() 
{
    fprintf(stderr, "RocprofDataSource::end\n");
    flush();
}

void RocprofDataSource::startTracing()
{
    fprintf(stderr, "RocprofDataSource::startTracing\n");
    if (client_ctx.handle != 0)
        rocprofiler_start_context(client_ctx);

}

void RocprofDataSource::stopTracing()
{
    fprintf(stderr, "RocprofDataSource::stopTracing\n");
    if (client_ctx.handle != 0)
        rocprofiler_stop_context(client_ctx);
}

void RocprofDataSource::flush()
{
    fprintf(stderr, "RocprofDataSource::flush\n");
    if (clientId != nullptr)
        rocprofiler_flush_buffer(client_buffer);
}

void RocprofDataSource::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    Logger &logger = Logger::singleton();

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        thread_local sqlite3_int64 timestamp;	// FIXME: use userdata?  or stack?

        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
                timestamp = clocktime_ns();
        }
        else {	     // ROCPROFILER_CALLBACK_PHASE_EXIT
            char buff[4096];
            ApiTable::row row;

            //const char *name = fmt::format("{}::{}", record.kind, record.operation).c_str();
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(std::string(name_info[record.kind][record.operation]).c_str());
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
}

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
                //sqlite3_int64 name_id = logger.stringTable().getOrCreate(fmt::format("{}::{}", record->kind, record->operation).c_str());
                // FIXME: op name hack
                static sqlite3_int64 name_id = logger.stringTable().getOrCreate("KernelExecution");
                sqlite3_int64 desc_id = logger.stringTable().getOrCreate(kernel_names.at(record->dispatch_info.kernel_id));

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
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {

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
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ;
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            kernel_info.emplace(data->kernel_id, *data);
            kernel_names.emplace(data->kernel_id, cxx_demangle(data->kernel_name));
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            kernel_info.erase(data->kernel_id);
            //kernel_names.erase(data->kernel_id);
        }
    }
}


// 
//
//

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    fprintf(stderr, "rocprofiler_configure\n");
    id->name = "rpd_tracer";
    clientId = id;

    // return pointer to configure data
    return &cfg;
}

int RocprofDataSource::toolInit(rocprofiler_client_finalize_t finialize_func, void* tool_data)
{
    fprintf(stderr, "RocprofDataSource::toolInit\n");

    name_info = common::get_buffer_tracing_names();

    rocprofiler_create_context(&client_ctx);

    rocprofiler_configure_callback_tracing_service(client_ctx,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                   nullptr,
                                                   0,
                                                   api_callback,
                                                   nullptr);
                                                   //tool_data);

// Magic stuff

    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    rocprofiler_configure_callback_tracing_service(client_ctx,
                                                   ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                   code_object_ops.data(),
                                                   code_object_ops.size(),
                                                   RocprofDataSource::code_object_callback,
                                                   nullptr);

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
                                                 ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                 nullptr,
                                                 0,
                                                 client_buffer);

    auto client_thread = rocprofiler_callback_thread_t{};
    rocprofiler_create_callback_thread(&client_thread);
    rocprofiler_assign_callback_thread(client_buffer, client_thread);

    int valid_ctx = 0;
    rocprofiler_context_is_valid(client_ctx, &valid_ctx);
    if (valid_ctx == 0) {
       client_ctx.handle = 0;	// Can't destroy it, so leak it
       return -1;
    }

    rocprofiler_start_context(client_ctx);
    return 0;
}

void RocprofDataSource::toolFinialize(void* tool_data)
{
    // This seems to happen pretty early.  So simulate a shutdown and disable context
    fprintf(stderr, "RocprofDataSource::toolFinalize\n");

    //end();	// FIXME: singleton - figure out startup order and teardown order

    // FIXME: kernel code objects are already being removed by this point
    //        keeping names (only) around to remedy this

    rocprofiler_stop_context(client_ctx);
    rocprofiler_flush_buffer(client_buffer);	// hopfully this blocks

    client_ctx.handle = 0;	// save us from ourselves
}
