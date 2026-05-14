/*********************************************************************************
* Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "RtlDataSource.h"

#include <sqlite3.h>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <dlfcn.h>

#include "Logger.h"
#include "Utility.h"

using rpdtracer::DataSource;
using rpdtracer::RtlDataSource;

extern "C" {
    DataSource *RtlDataSourceFactory() { return new RtlDataSource(); }
}

static std::once_flag register_once;
static std::once_flag registerAgain_once;

// ---- API event callback ----

void RtlDataSource::on_api_event(const trace_db::ApiEventRecord& event, void* user_data)
{
    Logger &logger = Logger::singleton();

    static sqlite3_int64 domain_id = logger.stringTable().getOrCreate("hip");

    sqlite3_int64 name_id = logger.stringTable().getOrCreate(event.name);

    ApiTable::row row;
    row.pid = event.pid;
    row.tid = event.tid;
    row.start = event.start_ns;
    row.end = event.end_ns;
    row.domain_id = domain_id;
    row.category_id = EMPTY_STRING_ID;
    row.apiName_id = name_id;
    row.api_id = event.correlation_id;

    if (event.args && event.args[0] != '\0') {
        row.args_id = logger.ustringTable().create(std::string(event.args));
    } else {
        row.args_id = EMPTY_STRING_ID;
    }

    if (strncmp(event.name, "hipModuleLaunchKernel", 21) == 0 ||
        strncmp(event.name, "hipExtModuleLaunchKernel", 24) == 0) {
        unsigned int gx = 0, gy = 0, gz = 0, bx = 0, by = 0, bz = 0, shared = 0;
        if (event.args) {
            sscanf(event.args, "grid=%u,%u,%u block=%u,%u,%u shared=%u",
                   &gx, &gy, &gz, &bx, &by, &bz, &shared);
        }

        KernelApiTable::row krow;
        krow.api_id = row.api_id;
        krow.gridX = gx;
        krow.gridY = gy;
        krow.gridZ = gz;
        krow.workgroupX = bx;
        krow.workgroupY = by;
        krow.workgroupZ = bz;
        krow.groupSegmentSize = shared;
        krow.privateSegmentSize = 0;
        krow.kernelName_id = EMPTY_STRING_ID;

        logger.kernelApiTable().insert(krow);
    }

    if (strncmp(event.name, "hipMemcpy", 9) == 0) {
        size_t sz = 0;
        int kind = 0;
        if (event.args) {
            sscanf(event.args, "size=%zu kind=%d", &sz, &kind);
        }

        CopyApiTable::row crow;
        crow.api_id = row.api_id;
        crow.size = (int)sz;
        crow.kind = kind;
        crow.sync = (strcmp(event.name, "hipMemcpy") == 0);

        logger.copyApiTable().insert(crow);
    }

    logger.apiTable().insert(row);

    std::call_once(register_once, atexit, Logger::rpdFinalize);
}

// ---- Kernel event callback ----

void RtlDataSource::on_kernel_event(const trace_db::KernelEventRecord& event, void* user_data)
{
    Logger &logger = Logger::singleton();

    sqlite3_int64 name_id = logger.stringTable().getOrCreate(event.name);
    sqlite3_int64 desc_id = logger.stringTable().getOrCreate(event.name);

    static sqlite3_int64 kernel_type_id = logger.stringTable().getOrCreate("KernelExecution");

    OpTable::row row;
    row.gpuId = event.device_id;
    row.queueId = (int)event.queue_id;
    row.sequenceId = 0;
    row.start = event.start_ns;
    row.end = event.end_ns;
    row.description_id = desc_id;
    row.opType_id = kernel_type_id;
    row.api_id = event.correlation_id;

    logger.opTable().insert(row);

    std::call_once(registerAgain_once, atexit, Logger::rpdFinalize);
}

// ---- DataSource lifecycle ----

void RtlDataSource::init()
{
    if (!getenv("HSA_TOOLS_LIB")) {
        Dl_info info;
        if (dladdr((void*)RtlDataSourceFactory, &info) && info.dli_fname) {
            setenv("HSA_TOOLS_LIB", info.dli_fname, 0);
        }
    }

    trace_db::set_api_event_callback(on_api_event, nullptr);
    trace_db::set_kernel_event_callback(on_kernel_event, nullptr);

    stopTracing();
}

void RtlDataSource::startTracing()
{
}

void RtlDataSource::stopTracing()
{
}

void RtlDataSource::flush()
{
}

void RtlDataSource::end()
{
    trace_db::rtl_trigger_shutdown();
}
