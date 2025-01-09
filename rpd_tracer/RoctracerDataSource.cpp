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
#include "RoctracerDataSource.h"

#include <roctracer_hip.h>
#include <roctracer_ext.h>
#include <roctracer_roctx.h>
#include <hsa/hsa_ext_amd.h>

#include <sqlite3.h>
#include <fmt/format.h>

#include <sstream>
#include <cpptrace/cpptrace.hpp>

#include "Logger.h"
#include "Utility.h"


// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *RoctracerDataSourceFactory() { return new RoctracerDataSource(); }
}  // extern "C"

// FIXME: can we avoid shutdown corruption?
// Other rocm libraries crashing on unload
// libsqlite unloading before we are done using it
// Current workaround: register an onexit function when first activity is delivered back
//                     this let's us unload first, or close to.
// New workaround: register 3 times, only finalize once.  see register_once

static std::once_flag register_once;
static std::once_flag registerAgain_once;

//RoctracerDataSource::RoctracerDataSource()
//{
//}

// Local copy of hip op types.  These are public (and stable) in later rocm releases
typedef enum {
  HIP_OP_COPY_KIND_UNKNOWN_ = 0,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_ = 0x11F3,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_ = 0x11F4,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_ = 0x11F5,
  HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_ = 0x1201,
  HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_ = 0x1202,
  HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_ = 0x1203,
  HIP_OP_COPY_KIND_FILL_BUFFER_ = 0x1207
} hip_op_copy_kind_t_;

typedef enum {
  HIP_OP_DISPATCH_KIND_UNKNOWN_ = 0,
  HIP_OP_DISPATCH_KIND_KERNEL_ = 0x11F0,
  HIP_OP_DISPATCH_KIND_TASK_ = 0x11F1
} hip_op_dispatch_kind_t_;

typedef enum {
  HIP_OP_BARRIER_KIND_UNKNOWN_ = 0
} hip_op_barrier_kind_t_;
// end hip op defines

namespace {
    int deviceOffset = 0x7fffffff;

    void createDeviceMap() {
        int dc = 0;
        hipGetDeviceCount(&dc);
        hsa_iterate_agents(
            [](hsa_agent_t agent, void* data) {
           auto &deviceOffset = *static_cast<int*>(data);
           int nodeId;
           hsa_agent_get_info(
                agent,
                static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                &nodeId);
           int deviceType;
           hsa_agent_get_info(
                agent,
                static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_DEVICE),
                &deviceType);
           if ((nodeId < deviceOffset) && (deviceType == HSA_DEVICE_TYPE_GPU))
               deviceOffset = nodeId;

           return HSA_STATUS_SUCCESS;
      } , &deviceOffset);
    };

    int mapDeviceId(int id) { return id - deviceOffset; };
} // namespace

#ifndef RPD_STACKFRAME_SUPPORT
int RoctracerDataSource::unwind(Logger &logger, const char *api, const sqlite_int64 api_id) {

     if (!logger.writeStackFrames()) return 0;

#if 0
     // for reference: full stack w/o manipulations
     const std::string stack1 = cpptrace::generate_trace(0).to_string(false);
     std::cout << stack1 << std::endl;
     if (true) return 0;
#endif

     // strip out the top frames that only point into roctracer/rpd, do not add color
     const std::string stack = cpptrace::generate_trace(3).to_string(false);
     /*
      * returns:
      * Stack trace (most recent call first):
      * #0 0x00007f3a1d2f4447 at /opt/rocm/lib/libamdhip64.so.6
      * #1 0x00000000002055cf in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:170:5
      * #2 0x00007f3a1cb561c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
      * #3 0x00007f3a1cb5628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
      * #4 0x0000000000204b04 at /root/rocm-examples/Applications/bitonic_sort/applications_bitonic_sort
      *
      * need to get rid of the first line
      * should inject api into #0 frame as "in $api at"
      */
     std::istringstream iss(stack);
     std::string line;
     std::getline(iss, line); // get rid of "Stack trace (most recent call first):"

     std::getline(iss, line);
     std::string s1 = line.substr(0,21);
     std::string s2 = line.substr(21);

     std::string fixed = s1 + " in " + api + "()" + s2;

     StackFrameTable::row frame0;
     frame0.api_id = api_id;
     frame0.depth = 0;
     frame0.name_id = logger.stringTable().getOrCreate(fixed.c_str());
     logger.stackFrameTable().insert(frame0);

     int n = 1;
     while ( std::getline(iss, line) ) {
         if (line.empty())
             continue;
	 StackFrameTable::row frame;
	 frame.api_id = api_id;
	 frame.depth = n;
	 frame.name_id = logger.stringTable().getOrCreate(line.c_str());
	 logger.stackFrameTable().insert(frame);

	 n++;
     }

     return 0;
}
#else
int RoctracerDataSource::unwind(Logger &logger, const char *api, const sqlite_int64 api_id) {return 0;}
#endif

void RoctracerDataSource::api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
    Logger &logger = Logger::singleton();

    if (domain == ACTIVITY_DOMAIN_HIP_API) {
        thread_local sqlite3_int64 timestamp;
        const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
            timestamp = clocktime_ns();
        }
        else { // data->phase == ACTIVITY_API_PHASE_EXIT
            char buff[4096];
            ApiTable::row row;

            const char *name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(name);
            row.pid = GetPid();
            row.tid = GetTid();
            row.start = timestamp;  // From TLS from preceding enter call
            row.end = clocktime_ns();
            row.apiName_id = name_id;
            row.args_id = EMPTY_STRING_ID;
            row.api_id = data->correlation_id;
#if 1
            switch (cid) {
                case HIP_API_ID_hipMalloc:
                    std::snprintf(buff, 4096, "ptr=%p | size=0x%x",
                        *data->args.hipMalloc.ptr,
                        (uint32_t)(data->args.hipMalloc.size));
                    row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    unwind(logger, "hipMalloc", row.api_id);
                    break;
                case HIP_API_ID_hipFree:
                    std::snprintf(buff, 4096, "ptr=%p",
                        data->args.hipFree.ptr);
                    row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    unwind(logger, "hipFree", row.api_id);
                    break;

                case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
                    {
                        const hipLaunchParams &params = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
                        std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.func, params.stream));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

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
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipLaunchCooperativeKernelMultiDevice", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
                    {
                        const hipLaunchParams &params = data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val;
                        std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.func, params.stream));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

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
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipExtLaunchMultiKernelMultiDevice", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipLaunchKernel:
                    {
                        auto &params = data->args.hipLaunchKernel;
                        std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.function_address, params.stream));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = params.numBlocks.x;
                        krow.gridY = params.numBlocks.y;
                        krow.gridZ = params.numBlocks.z;
                        krow.workgroupX = params.dimBlocks.x;
                        krow.workgroupY = params.dimBlocks.y;
                        krow.workgroupZ = params.dimBlocks.z;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipLaunchKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipExtLaunchKernel:
                    {
                        auto &params = data->args.hipExtLaunchKernel;
                        std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.function_address, params.stream));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = params.numBlocks.x;
                        krow.gridY = params.numBlocks.y;
                        krow.gridZ = params.numBlocks.z;
                        krow.workgroupX = params.dimBlocks.x;
                        krow.workgroupY = params.dimBlocks.y;
                        krow.workgroupZ = params.dimBlocks.z;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipExtLaunchKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipLaunchCooperativeKernel:
                    {
                        auto &params = data->args.hipLaunchCooperativeKernel;
                        std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.f, params.stream));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = params.gridDim.x;
                        krow.gridY = params.gridDim.y;
                        krow.gridZ = params.gridDim.z;
                        krow.workgroupX = params.blockDimX.x;
                        krow.workgroupY = params.blockDimX.y;
                        krow.workgroupZ = params.blockDimX.z;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipLaunchCooperativeKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipHccModuleLaunchKernel:
                    {
                        auto &params = data->args.hipHccModuleLaunchKernel;
                        std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.hStream);
                        krow.gridX = params.globalWorkSizeX;
                        krow.gridY = params.globalWorkSizeY;
                        krow.gridZ = params.globalWorkSizeZ;
                        krow.workgroupX = params.blockDimX;
                        krow.workgroupY = params.blockDimY;
                        krow.workgroupZ = params.blockDimZ;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipHccModuleLaunchKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipModuleLaunchKernel:
                    {
                        auto &params = data->args.hipModuleLaunchKernel;
                        std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = params.gridDimX;
                        krow.gridY = params.gridDimY;
                        krow.gridZ = params.gridDimZ;
                        krow.workgroupX = params.blockDimX;
                        krow.workgroupY = params.blockDimY;
                        krow.workgroupZ = params.blockDimZ;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipModuleLaunchKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;
#if 0
                case HIP_API_ID_hipGraphLaunch:
                    {
                        auto &params = data->args.hipGraphLaunch;
                        std::string kernelName(fmt::format("Graph Kernel ({})", (void*)params.graphExec));
                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.stream);
                        krow.gridX = 0;
                        krow.gridY = 0;
                        krow.gridZ = 0;
                        krow.workgroupX = 0;
                        krow.workgroupY = 0;
                        krow.workgroupZ = 0;
                        krow.groupSegmentSize = 0;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipGraphLaunch", row.api_id);

                        // Don't associate kernel name, would require a rework to support multiple
                        //   ops using the same entry.
                    }
                    break;
#endif

                case HIP_API_ID_hipExtModuleLaunchKernel:
                    {
                        auto &params = data->args.hipExtModuleLaunchKernel;
                        std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                        //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                        //    params.stream,
                        //    kernelName.c_str());
                        //row.args_id = logger.stringTable().getOrCreate(std::string(buff));

                        KernelApiTable::row krow;
                        krow.api_id = row.api_id;
                        krow.stream = fmt::format("{}", (void*)params.hStream);
                        krow.gridX = params.globalWorkSizeX;
                        krow.gridY = params.globalWorkSizeY;
                        krow.gridZ = params.globalWorkSizeZ;
                        krow.workgroupX = params.localWorkSizeX;
                        krow.workgroupY = params.localWorkSizeY;
                        krow.workgroupZ = params.localWorkSizeZ;
                        krow.groupSegmentSize = params.sharedMemBytes;
                        krow.privateSegmentSize = 0;
                        krow.kernelName_id = logger.stringTable().getOrCreate(kernelName);

                        logger.kernelApiTable().insert(krow);
                        unwind(logger, "hipExtModuleLaunchKernel", row.api_id);

                        // Associate kernel name with op
                        //logger.opTable().associateDescription(row.api_id, krow.kernelName_id);
                    }
                    break;

                case HIP_API_ID_hipMemcpy:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpy.dst,
                    //    data->args.hipMemcpy.src,
                    //    (uint32_t)(data->args.hipMemcpy.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpy.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpy.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpy.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpy.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpy.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpy", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpy2D:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | width=0x%x | height=0x%x | kind=%u",
                    //    data->args.hipMemcpy2D.dst,
                    //    data->args.hipMemcpy2D.src,
                    //    (uint32_t)(data->args.hipMemcpy2D.width),
                    //    (uint32_t)(data->args.hipMemcpy2D.height),
                    //    (uint32_t)(data->args.hipMemcpy2D.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.width = (uint32_t)(data->args.hipMemcpy2D.width);
                        crow.height = (uint32_t)(data->args.hipMemcpy2D.height);
                        crow.dst = fmt::format("{}", data->args.hipMemcpy2D.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpy2D.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpy2D.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpy2D", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpy2DAsync:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | width=0x%x | height=0x%x | kind=%u",
                    //    data->args.hipMemcpy2DAsync.dst,
                    //    data->args.hipMemcpy2DAsync.src,
                    //    (uint32_t)(data->args.hipMemcpy2DAsync.width),
                    //    (uint32_t)(data->args.hipMemcpy2DAsync.height),
                    //    (uint32_t)(data->args.hipMemcpy2DAsync.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpy2DAsync.stream);
                        crow.width = (uint32_t)(data->args.hipMemcpy2DAsync.width);
                        crow.height = (uint32_t)(data->args.hipMemcpy2DAsync.height);
                        crow.dst = fmt::format("{}", data->args.hipMemcpy2DAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpy2DAsync.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpy2DAsync.kind);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpy2DAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyAsync:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpyAsync.dst,
                    //    data->args.hipMemcpyAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyAsync.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyAsync.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyAsync.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpyAsync.kind);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyDtoD:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyDtoD.dst,
                    //    data->args.hipMemcpyDtoD.src,
                    //    (uint32_t)(data->args.hipMemcpyDtoD.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyDtoD.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyDtoD.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyDtoD.src);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyDtoD", row.api_id);
                    }

                    break;
                case HIP_API_ID_hipMemcpyDtoDAsync:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyDtoDAsync.dst,
                    //    data->args.hipMemcpyDtoDAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyDtoDAsync.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyDtoDAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyDtoDAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyDtoDAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyDtoDAsync.src);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyDtoDAsync", row.api_id);
                    }

                    break;
                case HIP_API_ID_hipMemcpyDtoH:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyDtoH.dst,
                    //    data->args.hipMemcpyDtoH.src,
                    //    (uint32_t)(data->args.hipMemcpyDtoH.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyDtoH.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyDtoH.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyDtoH.src);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyDtoH", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyDtoHAsync:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyDtoHAsync.dst,
                    //    data->args.hipMemcpyDtoHAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyDtoHAsync.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyDtoHAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyDtoHAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyDtoHAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyDtoHAsync.src);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyDtoHAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyFromSymbol:
                    //std::snprintf(buff, 4096, "dst=%p | symbol=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpyFromSymbol.dst,
                    //    data->args.hipMemcpyFromSymbol.symbol,
                    //    (uint32_t)(data->args.hipMemcpyFromSymbol.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyFromSymbol.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyFromSymbol.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyFromSymbol.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyFromSymbol.symbol);
                        crow.kind = (uint32_t)(data->args.hipMemcpyFromSymbol.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyFromSymbol", row.api_id);
                    }
                    break;
		case HIP_API_ID_hipMemcpyFromSymbolAsync:
                    //std::snprintf(buff, 4096, "dst=%p | symbol=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpyFromSymbolAsync.dst,
                    //    data->args.hipMemcpyFromSymbolAsync.symbol,
                    //    (uint32_t)(data->args.hipMemcpyFromSymbolAsync.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyFromSymbolAsync.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyFromSymbolAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyFromSymbolAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyFromSymbolAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyFromSymbolAsync.symbol);
                        crow.kind = (uint32_t)(data->args.hipMemcpyFromSymbolAsync.kind);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyFromSymbolAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyHtoD:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyHtoDAsync.dst,
                    //    data->args.hipMemcpyHtoDAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyHtoD.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyHtoD.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyHtoD.src);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyHtoD", row.api_id);
                    }
                    break;
		case HIP_API_ID_hipMemcpyHtoDAsync:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
                    //    data->args.hipMemcpyHtoDAsync.dst,
                    //    data->args.hipMemcpyHtoDAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyHtoDAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyHtoDAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyHtoDAsync.src);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyHtoDAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyPeer:
                    //std::snprintf(buff, 4096, "dst=%p | device=%d | src=%p | device=%d | size=0x%x",
                    //    data->args.hipMemcpyPeer.dst,
                    //    data->args.hipMemcpyPeer.dstDeviceId,
                    //    data->args.hipMemcpyPeer.src,
                    //    data->args.hipMemcpyPeer.srcDeviceId,
                    //    (uint32_t)(data->args.hipMemcpyPeer.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyPeer.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyPeer.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyPeer.src);
                        crow.dstDevice = data->args.hipMemcpyPeer.dstDeviceId;
                        crow.srcDevice = data->args.hipMemcpyPeer.srcDeviceId;
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyPeer", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyPeerAsync:
                    //std::snprintf(buff, 4096, "dst=%p | device=%d | src=%p | device=%d | size=0x%x",
                    //    data->args.hipMemcpyPeerAsync.dst,
                    //    data->args.hipMemcpyPeerAsync.dstDeviceId,
                    //    data->args.hipMemcpyPeerAsync.src,
                    //    data->args.hipMemcpyPeerAsync.srcDevice,
                    //    (uint32_t)(data->args.hipMemcpyPeerAsync.sizeBytes));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyPeerAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyPeerAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyPeerAsync.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyPeerAsync.src);
                        crow.dstDevice = data->args.hipMemcpyPeerAsync.dstDeviceId;
                        crow.srcDevice = data->args.hipMemcpyPeerAsync.srcDevice;
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyPeerAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyToSymbol:
                    //std::snprintf(buff, 4096, "symbol=%p | src=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpyToSymbol.symbol,
                    //    data->args.hipMemcpyToSymbol.src,
                    //    (uint32_t)(data->args.hipMemcpyToSymbol.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyToSymbol.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.size = (uint32_t)(data->args.hipMemcpyToSymbol.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyToSymbol.symbol);
                        crow.src = fmt::format("{}", data->args.hipMemcpyToSymbol.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpyToSymbol.kind);
                        crow.sync = true;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyToSymbol", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyToSymbolAsync:
                    //std::snprintf(buff, 4096, "symbol=%p | src=%p | size=0x%x | kind=%u",
                    //    data->args.hipMemcpyToSymbolAsync.symbol,
                    //    data->args.hipMemcpyToSymbolAsync.src,
                    //    (uint32_t)(data->args.hipMemcpyToSymbolAsync.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyToSymbolAsync.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff));
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyToSymbolAsync.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyToSymbolAsync.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyToSymbolAsync.symbol);
                        crow.src = fmt::format("{}", data->args.hipMemcpyToSymbolAsync.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpyToSymbolAsync.kind);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyToSymbolAsync", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipMemcpyWithStream:
                    //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u", 
                    //    data->args.hipMemcpyWithStream.dst,
                    //    data->args.hipMemcpyWithStream.src,
                    //    (uint32_t)(data->args.hipMemcpyWithStream.sizeBytes),
                    //    (uint32_t)(data->args.hipMemcpyWithStream.kind));
                    //row.args_id = logger.stringTable().getOrCreate(std::string(buff)); 
                    {
                        CopyApiTable::row crow;
                        crow.api_id = row.api_id;
                        crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyWithStream.stream);
                        crow.size = (uint32_t)(data->args.hipMemcpyWithStream.sizeBytes);
                        crow.dst = fmt::format("{}", data->args.hipMemcpyWithStream.dst);
                        crow.src = fmt::format("{}", data->args.hipMemcpyWithStream.src);
                        crow.kind = (uint32_t)(data->args.hipMemcpyWithStream.kind);
                        crow.sync = false;
                        logger.copyApiTable().insert(crow);
                        unwind(logger, "hipMemcpyWithStream", row.api_id);
                    }
                    break;
                case HIP_API_ID_hipStreamBeginCapture:
                    row.args_id = logger.stringTable().getOrCreate(
                        fmt::format("stream = {} | mode = {}", (void*)data->args.hipStreamBeginCapture.stream, data->args.hipStreamBeginCapture.mode)
                    );
                    unwind(logger, name, row.api_id);
                    break;
                case HIP_API_ID_hipStreamEndCapture:
                    row.args_id = logger.stringTable().getOrCreate(
                        fmt::format("stream = {} | graph = {}", (void*)data->args.hipStreamEndCapture.stream, (void*)*(data->args.hipStreamEndCapture.pGraph))
                    );
                    unwind(logger, name, row.api_id);
                    break;
                case HIP_API_ID_hipGraphInstantiate:
                    row.args_id = logger.stringTable().getOrCreate(
                        fmt::format("graphExec = {} | graph = {}", (void *)*(data->args.hipGraphInstantiate.pGraphExec), (void *)data->args.hipGraphInstantiate.graph)
                    );
                    unwind(logger, name, row.api_id);
                    break;
                case HIP_API_ID_hipGraphInstantiateWithFlags:
                    row.args_id = logger.stringTable().getOrCreate(
                        fmt::format("graphExec = {} | graph = {}", (void *)*(data->args.hipGraphInstantiateWithFlags.pGraphExec), (void *)data->args.hipGraphInstantiateWithFlags.graph)
                    );
                    unwind(logger, name, row.api_id);
                    break;
                case HIP_API_ID_hipGraphLaunch:
                    row.args_id = logger.stringTable().getOrCreate(
                        fmt::format("graphExec = {} | stream = {}", (void *)data->args.hipGraphLaunch.graphExec, (void *)data->args.hipGraphLaunch.stream)
                    );
                    unwind(logger, name, row.api_id);
                    break;
                default:
                    unwind(logger, name, row.api_id);
                    break;
            }
#endif
            logger.apiTable().insert(row);
        }
    }

    if (domain == ACTIVITY_DOMAIN_ROCTX) {
        const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);

        ApiTable::row row;
        row.pid = GetPid();
        row.tid = GetTid();
        row.start = clocktime_ns();
        row.end = row.start;
        static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
        row.apiName_id = markerId;
        row.args_id = EMPTY_STRING_ID;
        row.api_id = 0;

        switch (cid) {
            case ROCTX_API_ID_roctxMarkA:
                row.args_id = logger.stringTable().getOrCreate(data->args.message);
                logger.apiTable().insertRoctx(row);
                break;
            case ROCTX_API_ID_roctxRangePushA:
                row.args_id = logger.stringTable().getOrCreate(data->args.message);
                logger.apiTable().pushRoctx(row);
                break;
            case ROCTX_API_ID_roctxRangePop:
                logger.apiTable().popRoctx(row);
                break;
            default:
                break;
        }
    }
    std::call_once(register_once, atexit, Logger::rpdFinalize);
}


#if 0
void hip_activity_callback(const char* begin, const char* end, void* arg)
{
return;
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
    const timestamp_t cb_begin_time = clocktime_ns();

    int batchSize = 0;

    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    while (record < end_record) {
        if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        int index = 0;
        sqlite_int64 rowId = 0;
        int ret = 0;
//printf("hip: %s\n", name);

        if ((record->op != HIP_API_ID_hipGetDevice) && (record->op != HIP_API_ID_hipSetDevice)) {
            //"insert into rocpd_string(string) values (?)"
            sqlite3_bind_text(stringInsert, 1, name, -1, SQLITE_STATIC);
            ret = sqlite3_step(stringInsert);
            sqlite3_reset(stringInsert);
            rowId = sqlite3_last_insert_rowid(connection);

            // "insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)"
            index = 1;
            sqlite3_bind_int(apiInsert, index++, record->correlation_id);
            sqlite3_bind_int(apiInsert, index++, record->process_id);
            sqlite3_bind_int(apiInsert, index++, record->thread_id);
            sqlite3_bind_int64(apiInsert, index++, record->begin_ns);
            sqlite3_bind_int64(apiInsert, index++, record->end_ns);
            sqlite3_bind_int64(apiInsert, index++, rowId);
            sqlite3_bind_int64(apiInsert, index++, EMPTY_STRING_ID);

            ret = sqlite3_step(apiInsert);
            sqlite3_reset(apiInsert);
        }
        }
        roctracer_next_record(record, &record);
        ++batchSize;
    }
    const timestamp_t cb_mid_time = clocktime_ns();
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = clocktime_ns();
    printf("### activity_callback hip ### tid=%d ### %d (%d) %lu \n", GetTid(), count++, batchSize, (cb_end_time - cb_begin_time)/1000);

    // Make a tracer overhead record
    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);
    create_overhead_record("overhead (hip)", cb_begin_time, cb_end_time);
    create_overhead_record("prepare", cb_begin_time, cb_mid_time);
    create_overhead_record("commit", cb_mid_time, cb_end_time);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
}
#endif

void RoctracerDataSource::hcc_activity_callback(const char* begin, const char* end, void* arg)
{
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
    const timestamp_t cb_begin_time = clocktime_ns();

    int batchSize = 0;

    // Roctracer uses CLOCK_MONOTONIC for timestamps, which matches our timestamps.
    // However, the Roctracer developer thinks it is using CLOCK_MONOTONIC_RAW, which it isn't
    // Go ahead and convert timestamps here just in case this gets "fixed" at some point
    timestamp_t t0, t1, t00;
    roctracer_get_timestamp(&t1);	// first call is really slow, throw it away
    t0 = clocktime_ns();
    roctracer_get_timestamp(&t1);
    t00 = clocktime_ns();
    const timestamp_t toffset = (t0 >> 1) + (t00 >> 1) - t1;

    Logger &logger = Logger::singleton();

    while (record < end_record) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        if (record->op != HIP_OP_ID_BARRIER) { // Don't log markers
            sqlite3_int64 name_id = logger.stringTable().getOrCreate(name);

            OpTable::row row;
            row.gpuId = mapDeviceId(record->device_id);
            row.queueId = record->queue_id;
            row.sequenceId = 0;
            strncpy(row.completionSignal, "", 18);
            row.start = record->begin_ns + toffset;
            row.end = record->end_ns + toffset;
            row.description_id = ((record->kind == HIP_OP_DISPATCH_KIND_KERNEL_)
                               || (record->kind == HIP_OP_DISPATCH_KIND_TASK_))
                ? logger.stringTable().getOrCreate(cxx_demangle(record->kernel_name))
                : EMPTY_STRING_ID;
            row.opType_id = name_id;
            row.api_id = record->correlation_id;

            logger.opTable().insert(row);
        }
        roctracer_next_record(record, &record);
        ++batchSize;
    }
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d", batchSize);
    logger.createOverheadRecord(cb_begin_time, cb_end_time, "hcc_activity_callback", buff);

    std::call_once(registerAgain_once, atexit, Logger::rpdFinalize);
}



void RoctracerDataSource::init() {

    // Pick some apis to ignore
    m_apiList.setInvertMode(true);  // Omit the specified api
    m_apiList.add("hipGetDevice");
    m_apiList.add("hipSetDevice");
    m_apiList.add("hipGetLastError");
    m_apiList.add("__hipPushCallConfiguration");
    m_apiList.add("__hipPopCallConfiguration");
    m_apiList.add("hipCtxSetCurrent");
    m_apiList.add("hipEventRecord");
    m_apiList.add("hipEventQuery");
    m_apiList.add("hipGetDeviceProperties");
    m_apiList.add("hipPeekAtLastError");
    m_apiList.add("hipModuleGetFunction");
    m_apiList.add("hipEventCreateWithFlags");

    // roctracer properties
    //    Whatever the hell that means.  Magic encantation, thanks.
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);

    // Enable API callbacks
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL);

    if (m_apiList.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);
        const std::unordered_map<uint32_t, uint32_t> &filter = m_apiList.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        const std::unordered_map<uint32_t, uint32_t> &filter = m_apiList.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, NULL);
        }
    }

#if 1 
    // Work around a roctracer bug.  Must have a default pool or crash at exit
    // Allocating tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);
#endif

#if 1
    // Log hcc
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    //hcc_cb_properties.buffer_size = 0x1000; //0x40000;
    hcc_cb_properties.buffer_size = 0x40000;
    hcc_cb_properties.buffer_callback_fun = hcc_activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &m_hccPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, m_hccPool);
#endif

    createDeviceMap();
    stopTracing();
}

void RoctracerDataSource::startTracing() {
    //printf("# START ############################# %d\n", GetTid());
    roctracer_start();
}

void RoctracerDataSource::stopTracing() {
    //printf("# STOP #############################\n");
    roctracer_stop();
}

void RoctracerDataSource::flush() {
    roctracer_flush_activity();
    roctracer_flush_activity_expl(m_hccPool);
}

void RoctracerDataSource::end() {
    roctracer_stop();
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS);

    roctracer_flush_activity();
    roctracer_flush_activity_expl(m_hccPool);
    m_hccPool = nullptr;
}

uint32_t RocmApiIdList::mapName(const std::string &apiName)
{
    uint32_t cid = 0;
    if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
        return cid;
    else
        return 0;
}
