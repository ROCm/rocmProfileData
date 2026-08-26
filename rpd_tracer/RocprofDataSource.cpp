// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "RocprofDataSource.h"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>

#include <vector>
#include <array>
#include <set>
#include <string>

#include <sqlite3.h>
#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include "Logger.h"
#include "LocalStringCache.h"
#include "UStringCache.h"
#include "Utility.h"

using rpdtracer::DataSource;
using rpdtracer::RocprofDataSource;
//using rpdtracer::RocprofApiIdList;


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

                auto &crow = *(static_cast<rpdtracer::CopyApiTable::row*>(cb_data));
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
                    auto &krow = *(static_cast<rpdtracer::KernelApiTable::row*>(cb_data));
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

    // Extract hip args to json
            auto extract_hip_args = [](rocprofiler_buffer_tracing_kind_t,
                  rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   void*             cb_data) -> int {
                nlohmann::json &json = *(static_cast<nlohmann::json*>(cb_data));
                json[arg_name] = arg_value_str;
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

    class RocprofApiIdList : public rpdtracer::ApiIdList
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
    rocprofiler_client_finalize_t finalizer {nullptr};
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

    kernel_name_map_t kernel_names = {};

    // Manage buffer name - #betterThanRoctracer
    buffer_name_info name_info = {};

    // Agent info
    // <rocprofiler_profile_config_id_t.handle, rocprofiler_agent_v0_t>
    agent_info_map_t agents = {};

    // ---- Counter collection state ----
    bool collectCounters {false};

    // Counter sets: each set is a group of counter names collected together.
    // Dispatches round-robin through sets per kernel name.
    std::vector<std::set<std::string>> counterSets;

    // Cached valid counter configs per agent: [agent_handle] → config_ids
    // Built lazily on first dispatch per agent. Only contains configs that
    // were successfully created (handle != 0).
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>> counterConfigs;
    std::mutex counterConfigMutex;

    // Counter name info: counter_id.handle → name string (from SDK)
    std::unordered_map<uint64_t, std::string> counterIdNames;

    // Per-kernel-name dispatch count for RR set selection.
    // Keyed on name (not kernel_id) because the same kernel loaded from
    // different code objects gets distinct kernel_ids.
    std::unordered_map<std::string, uint64_t> kernelDispatchCount;

    // kernel_id → kernel name for RR lookup.  Populated at code-object-load
    // time so the dispatch callback never touches kernel_names (no lock).
    std::unordered_map<uint64_t, std::string> kernelIdToName;
    std::mutex kernelDispatchMutex;

    // dispatch_id → op_id mapping for counter support
    // Populated from kernel dispatch records in buffer_callback,
    // consumed by counter_buffer_callback to link counters to ops.
    std::unordered_map<uint64_t, sqlite3_int64> dispatchOpId;

    // Counters whose dimension instances should be averaged (not summed).
    // Utilization/percentage metrics — the smaller set.
    static const std::set<std::string>& averagedCounters() {
        static const std::set<std::string> s = {
            "VALUBusy",
            "SALUBusy",
            "MemUnitBusy",
            "MemUnitStalled",
            "VALUUtilization",
            "FetchSize",       // already normalized per-SE by HW
            "WriteSize",
            "L2CacheHit",
            "WriteUnitStalled",
        };
        return s;
    }

    void buildCounterConfigs(rocprofiler_agent_id_t agent_id);

private:
    RocprofDataSourceShared() { s = this; }
    ~RocprofDataSourceShared() { s = nullptr; }
};

RocprofDataSourceShared &RocprofDataSourceShared::singleton()
{
    static RocprofDataSourceShared *instance = new RocprofDataSourceShared();	// Leak this
    return *instance;
}

void RocprofDataSourceShared::buildCounterConfigs(rocprofiler_agent_id_t agent_id)
{
    std::lock_guard<std::mutex> lock(counterConfigMutex);
    if (counterConfigs.count(agent_id.handle))
        return;

    // Enumerate all counters supported by this agent
    std::vector<rocprofiler_counter_id_t> gpu_counters;
    rocprofiler_iterate_agent_supported_counters(
        agent_id,
        [](rocprofiler_agent_id_t,
           rocprofiler_counter_id_t* counters,
           size_t num_counters,
           void* user_data) {
            auto* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            for (size_t i = 0; i < num_counters; i++)
                vec->push_back(counters[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        static_cast<void*>(&gpu_counters));

    // Build name→counter_id map and cache names
    std::unordered_map<std::string, rocprofiler_counter_id_t> nameToId;
    for (auto& counter : gpu_counters) {
        rocprofiler_counter_info_v0_t info;
        if (rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info))
            == ROCPROFILER_STATUS_SUCCESS) {
            nameToId[info.name] = counter;
            counterIdNames[counter.handle] = info.name;
        }
    }

    // Build a counter_config for each counter set, keeping only valid ones
    auto& configs = counterConfigs[agent_id.handle];
    for (size_t si = 0; si < counterSets.size(); ++si) {
        auto& counterSet = counterSets[si];
        std::vector<rocprofiler_counter_id_t> ids;
        for (auto& name : counterSet) {
            auto it = nameToId.find(name);
            if (it != nameToId.end())
                ids.push_back(it->second);
        }
        if (ids.empty()) {
            rpdtracer::rpdLog("rpd_tracer: counter set %ld has no supported counters for agent %lu, skipping\n",
                   si, agent_id.handle);
            continue;
        }
        rocprofiler_counter_config_id_t config = {.handle = 0};
        auto status = rocprofiler_create_counter_config(
            agent_id, ids.data(), ids.size(), &config);
        if (status != ROCPROFILER_STATUS_SUCCESS) {
            rpdtracer::rpdLog("rpd_tracer: counter config creation failed (status=%d) for agent %lu set %ld, skipping\n",
                   status, agent_id.handle, si);
            continue;
        }
        configs.push_back(config);
    }
}


namespace rpdtracer {

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

    bool logArgs { true };

    bool idsCached {false};
    sqlite3_int64 kernelExecId {0};
    sqlite3_int64 memcpyId {0};
    sqlite3_int64 domainId {0};
    sqlite3_int64 scratchDomainId {0};
    sqlite3_int64 kfdPageMigrateId {0};
    sqlite3_int64 kfdPageFaultId {0};
    sqlite3_int64 kfdQueueId {0};
    void cacheIds();
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

    // Suppress args logging
    d->logArgs = (atoi(getConfig("RPDT_ROCPROF_NOARGS", "rocprof_noargs", "0")) == 0);

    // Counter collection config is read in toolInit (runs before constructor completes)
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

    if (s != nullptr && s->finalizer != nullptr && s->clientId != nullptr) {
        s->finalizer(*s->clientId);
        s->finalizer = nullptr;
        s->clientId = nullptr;
    }
}

void RocprofDataSource::startTracing()
{
    if (s->contexts[d->id].handle == 0)
        return;
    rocprofiler_start_context(s->contexts[d->id]);
}

void RocprofDataSource::stopTracing()
{
    if (s->contexts[d->id].handle == 0)
        return;
    rocprofiler_stop_context(s->contexts[d->id]);
}

void RocprofDataSource::flush()
{
    rocprofiler_flush_buffer(s->client_buffers[d->id]);
}

void RocprofDataSourcePrivate::cacheIds()
{
    if (idsCached)
        return;
    Logger &logger = Logger::singleton();
    kernelExecId = logger.stringTable().getOrCreate("KernelExecution");
    memcpyId = logger.stringTable().getOrCreate("Memcpy");
    domainId = logger.stringTable().getOrCreate("hip");
    scratchDomainId = logger.stringTable().getOrCreate("scratch");
    kfdPageMigrateId = logger.stringTable().getOrCreate("kfd_page_migrate");
    kfdPageFaultId = logger.stringTable().getOrCreate("kfd_page_fault");
    kfdQueueId = logger.stringTable().getOrCreate("kfd_queue");
    idsCached = true;
}

void RocprofDataSource::reset()
{
    d->idsCached = false;
}


void RocprofDataSource::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    static thread_local rpdtracer::LocalStringCache t_stringCache;
    RocprofDataSource &instance = **(reinterpret_cast<RocprofDataSource**>(callback_data));
    instance.d->cacheIds();

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
            sqlite3_int64 name_id = t_stringCache.lookup(std::string(s->name_info[record.kind][record.operation]).c_str(), logger.stringTable(), logger.storageGeneration());
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
            krow.kernelName_id = t_stringCache.lookup(s->kernel_names.at(info.kernel_id), logger.stringTable(), logger.storageGeneration());

            logger.kernelApiTable().insert(krow);
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
            // completion callback - runtime thread
            auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
            auto &info = dispatch.dispatch_info;

            OpTable::row row;
            row.gpuId = s->agents.at(info.agent_id.handle).logical_node_type_id;
            row.queueId = info.queue_id.handle;
            row.sequenceId = info.dispatch_id;
            strncpy(row.completionSignal, "", 18);
            row.start = adjust_external_ts(dispatch.start_timestamp);
            row.end = adjust_external_ts(dispatch.end_timestamp);
            row.description_id = t_stringCache.lookup(s->kernel_names.at(info.kernel_id), logger.stringTable(), logger.storageGeneration());
            row.opType_id = instance.d->kernelExecId;
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

            OpTable::row row;
            //row.gpuId = mapDeviceId(record->device_id);
            row.gpuId = 0;	// FIXME intercept hsa to figure out node?
            row.queueId = 0;
            row.sequenceId = 0;
            strncpy(row.completionSignal, "", 18);
            row.start = adjust_external_ts(copy.start_timestamp);
            row.end = adjust_external_ts(copy.end_timestamp);
            row.description_id = t_stringCache.lookup(crow.kindStr, logger.stringTable(), logger.storageGeneration());
            row.opType_id = instance.d->memcpyId;
            row.api_id = record.correlation_id.internal;
            logger.opTable().insert(row);

            // dispose the copyapi row
            //instance.d->copyrows.erase(record.correlation_id.internal);
            // FIXME can not dispose after use.  Copyapi -> copyop can be 1:n
        }
    }
}
#endif

// roctx handling moved to RoctxDataSource


#if 1
void RocprofDataSource::buffer_callback(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id, rocprofiler_record_header_t** headers, size_t num_headers, void* user_data, uint64_t drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");
    static thread_local rpdtracer::LocalStringCache t_stringCache;
    RocprofDataSource &instance = **(reinterpret_cast<RocprofDataSource**>(user_data));
    instance.d->cacheIds();

    Logger &logger = Logger::singleton();

    int64_t last_correlation = -1;
    const timestamp_t cb_begin_time = clocktime_ns();

    // Counter accumulator state (for interleaved counter records in shared buffer)
    struct CounterAccum { double sum {0.0}; uint64_t count {0}; };
    constexpr size_t CTR_ACCUM_BUCKETS = 16;
    std::array<std::pair<uint64_t, CounterAccum>, CTR_ACCUM_BUCKETS> ctr_accum_slots{};
    size_t ctr_accum_used = 0;
    sqlite3_int64 ctr_op_id = 0;
    bool have_ctr_dispatch = false;

    auto flush_counters = [&]() {
        if (!have_ctr_dispatch)
            return;
        for (size_t j = 0; j < ctr_accum_used; ++j) {
            auto& [counter_handle, accum] = ctr_accum_slots[j];
            std::string counterName;
            {
                std::lock_guard<std::mutex> lock(s->counterConfigMutex);
                auto name_it = s->counterIdNames.find(counter_handle);
                if (name_it == s->counterIdNames.end())
                    continue;
                counterName = name_it->second;
            }
            bool shouldAverage = s->averagedCounters().count(counterName) > 0;
            double value = shouldAverage ? (accum.sum / accum.count) : accum.sum;
            sqlite3_int64 name_id = t_stringCache.lookup(counterName, logger.stringTable(), logger.storageGeneration());
            CounterTable::row row;
            row.op_id = ctr_op_id;
            row.name_id = name_id;
            row.value = value;
            logger.counterTable().insert(row);
        }
        ctr_accum_used = 0;
        have_ctr_dispatch = false;
    };

    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];

        if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING) {
            if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {

                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
                auto& dispatch = record->dispatch_info;
                sqlite3_int64 desc_id = t_stringCache.lookup(s->kernel_names.at(record->dispatch_info.kernel_id), logger.stringTable(), logger.storageGeneration());

                OpTable::row row;
                row.gpuId = s->agents.at(dispatch.agent_id.handle).logical_node_type_id;
                row.queueId = dispatch.queue_id.handle;
                row.sequenceId = 0;
                row.start = adjust_external_ts(record->start_timestamp);
                row.end = adjust_external_ts(record->end_timestamp);
                row.description_id = desc_id;
                row.opType_id = instance.d->kernelExecId;
                row.api_id = record->correlation_id.internal;

                sqlite3_int64 op_id = logger.opTable().insert(row);
                if (s->collectCounters)
                    s->dispatchOpId[dispatch.dispatch_id] = op_id;

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
                last_correlation = record->correlation_id.internal;
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {

                auto &copy = *(static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload));
                std::string op_name = std::string(s->name_info[copy.kind][copy.operation]);
                sqlite3_int64 name_id = t_stringCache.lookup(op_name.c_str(), logger.stringTable(), logger.storageGeneration());
                sqlite3_int64 desc_id = t_stringCache.lookup(op_name.c_str(), logger.stringTable(), logger.storageGeneration());

                auto &dst_agent = s->agents.at(copy.dst_agent_id.handle);
                auto &src_agent = s->agents.at(copy.src_agent_id.handle);

                // Add the op entry
                OpTable::row row;
                row.gpuId = (dst_agent.type == ROCPROFILER_AGENT_TYPE_GPU)
                    ? dst_agent.logical_node_type_id
                    : src_agent.logical_node_type_id;
                row.queueId = 0;
                row.sequenceId = 0;
                row.start = adjust_external_ts(copy.start_timestamp);
                row.end = adjust_external_ts(copy.end_timestamp);
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
                last_correlation = copy.correlation_id.internal;
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT) {
                auto &hipapi = *(static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(header->payload));

                // extract args as json
                nlohmann::json json;
                if (instance.d->logArgs) {
                    rocprofiler_iterate_buffer_tracing_record_args(
                        *header, extract_hip_args,
                        &json);
                }

                // Add an api table entry
                sqlite3_int64 name_id = t_stringCache.lookup(std::string(s->name_info[hipapi.kind][hipapi.operation]).c_str(), logger.stringTable(), logger.storageGeneration());

                ApiTable::row row;
                row.pid = GetPid();
                row.tid = hipapi.thread_id;
                row.start = adjust_external_ts(hipapi.start_timestamp);
                row.end = adjust_external_ts(hipapi.end_timestamp);
                row.domain_id = instance.d->domainId;
                row.category_id = EMPTY_STRING_ID;
                row.apiName_id = name_id;
                if (instance.d->logArgs) {
                    static thread_local rpdtracer::UStringCache t_ustringCache;
                    row.args_id = t_ustringCache.lookup(json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), logger.ustringTable(), logger.storageGeneration());
                }
                else
                    row.args_id = EMPTY_STRING_ID;
                row.api_id = hipapi.correlation_id.internal;

                logger.apiTable().insert(row);
                last_correlation = hipapi.correlation_id.internal;
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY) {
                auto &scratch = *(static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(header->payload));
                sqlite3_int64 name_id = t_stringCache.lookup(std::string(s->name_info[scratch.kind][scratch.operation]).c_str(), logger.stringTable(), logger.storageGeneration());

                ApiTable::row row;
                row.pid = GetPid();
                row.tid = scratch.thread_id;
                row.start = adjust_external_ts(scratch.start_timestamp);
                row.end = adjust_external_ts(scratch.end_timestamp);
                row.domain_id = instance.d->scratchDomainId;
                row.category_id = EMPTY_STRING_ID;
                row.apiName_id = name_id;
                if (instance.d->logArgs) {
                    static thread_local rpdtracer::UStringCache t_ustringCache;
                    nlohmann::json json;
                    json["size"] = scratch.allocation_size;
                    json["gpu"] = s->agents.at(scratch.agent_id.handle).logical_node_type_id;
                    json["queue"] = scratch.queue_id.handle;
                    json["flags"] = static_cast<int>(scratch.flags);
                    row.args_id = t_ustringCache.lookup(json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), logger.ustringTable(), logger.storageGeneration());
                }
                else
                    row.args_id = EMPTY_STRING_ID;
                row.api_id = scratch.correlation_id.internal;
                logger.apiTable().insert(row);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE) {
                auto &record = *(static_cast<rocprofiler_buffer_tracing_kfd_page_migrate_record_t*>(header->payload));
                auto op_name = std::string(s->name_info[record.kind][record.operation]);
                constexpr auto prefix_len = sizeof("ROCPROFILER_KFD_PAGE_MIGRATE_") - 1;
                if (op_name.size() > prefix_len) op_name = op_name.substr(prefix_len);
                sqlite3_int64 desc_id = t_stringCache.lookup(op_name.c_str(), logger.stringTable(), logger.storageGeneration());

                int gpuId = 0;
                if (s->agents.count(record.dst_agent.handle) && s->agents.at(record.dst_agent.handle).type == ROCPROFILER_AGENT_TYPE_GPU)
                    gpuId = s->agents.at(record.dst_agent.handle).logical_node_type_id;
                else if (s->agents.count(record.src_agent.handle) && s->agents.at(record.src_agent.handle).type == ROCPROFILER_AGENT_TYPE_GPU)
                    gpuId = s->agents.at(record.src_agent.handle).logical_node_type_id;

                OpTable::row row;
                row.gpuId = gpuId;
                row.queueId = -1;
                row.sequenceId = 0;
                row.start = adjust_external_ts(record.start_timestamp);
                row.end = adjust_external_ts(record.end_timestamp);
                row.description_id = desc_id;
                row.opType_id = instance.d->kfdPageMigrateId;
                row.api_id = 0;
                logger.opTable().insert(row);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT) {
                auto &record = *(static_cast<rocprofiler_buffer_tracing_kfd_page_fault_record_t*>(header->payload));
                auto op_name = std::string(s->name_info[record.kind][record.operation]);
                constexpr auto prefix_len = sizeof("ROCPROFILER_KFD_PAGE_FAULT_") - 1;
                if (op_name.size() > prefix_len) op_name = op_name.substr(prefix_len);
                sqlite3_int64 desc_id = t_stringCache.lookup(op_name.c_str(), logger.stringTable(), logger.storageGeneration());

                int gpuId = 0;
                if (s->agents.count(record.agent_id.handle))
                    gpuId = s->agents.at(record.agent_id.handle).logical_node_type_id;

                OpTable::row row;
                row.gpuId = gpuId;
                row.queueId = -2;
                row.sequenceId = 0;
                row.start = adjust_external_ts(record.start_timestamp);
                row.end = adjust_external_ts(record.end_timestamp);
                row.description_id = desc_id;
                row.opType_id = instance.d->kfdPageFaultId;
                row.api_id = 0;
                logger.opTable().insert(row);
            }
            else if (header->kind == ROCPROFILER_BUFFER_TRACING_KFD_QUEUE) {
                auto &record = *(static_cast<rocprofiler_buffer_tracing_kfd_queue_record_t*>(header->payload));
                auto op_name = std::string(s->name_info[record.kind][record.operation]);
                constexpr auto prefix_len = sizeof("ROCPROFILER_KFD_QUEUE_") - 1;
                if (op_name.size() > prefix_len) op_name = op_name.substr(prefix_len);
                sqlite3_int64 desc_id = t_stringCache.lookup(op_name.c_str(), logger.stringTable(), logger.storageGeneration());

                int gpuId = 0;
                if (s->agents.count(record.agent_id.handle))
                    gpuId = s->agents.at(record.agent_id.handle).logical_node_type_id;

                OpTable::row row;
                row.gpuId = gpuId;
                row.queueId = -3;
                row.sequenceId = 0;
                row.start = adjust_external_ts(record.start_timestamp);
                row.end = adjust_external_ts(record.end_timestamp);
                row.description_id = desc_id;
                row.opType_id = instance.d->kfdQueueId;
                row.api_id = 0;
                logger.opTable().insert(row);
            }
        }
        else if (header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS) {
            if (header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER) {
                flush_counters();

                auto* record = static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
                auto op_it = s->dispatchOpId.find(record->dispatch_info.dispatch_id);
                if (op_it != s->dispatchOpId.end()) {
                    ctr_op_id = op_it->second;
                    have_ctr_dispatch = true;
                    s->dispatchOpId.erase(op_it);
                }
            }
            else if (header->kind == ROCPROFILER_COUNTER_RECORD_VALUE && have_ctr_dispatch) {
                auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
                rocprofiler_counter_id_t counter_id = {.handle = 0};
                rocprofiler_query_record_counter_id(record->id, &counter_id);

                size_t slot = ctr_accum_used;
                for (size_t j = 0; j < ctr_accum_used; ++j) {
                    if (ctr_accum_slots[j].first == counter_id.handle) {
                        slot = j;
                        break;
                    }
                }
                if (slot == ctr_accum_used && ctr_accum_used < CTR_ACCUM_BUCKETS) {
                    ctr_accum_slots[ctr_accum_used] = {counter_id.handle, {0.0, 0}};
                    ctr_accum_used++;
                }
                if (slot < CTR_ACCUM_BUCKETS) {
                    ctr_accum_slots[slot].second.sum += record->counter_value;
                    ctr_accum_slots[slot].second.count++;
                }
            }
        }
    }
    flush_counters();

    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%ld last=%ld", num_headers, last_correlation);
    logger.createOverheadRecord(cb_begin_time, cb_end_time, "RocprofDataSource::buffer_callback", buff);
}
#endif

void RocprofDataSource::counter_dispatch_callback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_config_id_t* config,
    rocprofiler_user_data_t* user_data,
    void* callback_data)
{
    auto agent_id = dispatch_data.dispatch_info.agent_id;
    auto kernel_id = dispatch_data.dispatch_info.kernel_id;

    // Lazily build counter configs for this agent
    s->buildCounterConfigs(agent_id);

    std::vector<rocprofiler_counter_config_id_t> configs;
    {
        std::lock_guard<std::mutex> lock(s->counterConfigMutex);
        auto it = s->counterConfigs.find(agent_id.handle);
        if (it == s->counterConfigs.end())
            return;
        configs = it->second;
    }
    if (configs.empty())
        return;

    // RR: pick set based on per-kernel-name dispatch count
    uint64_t count;
    {
        std::lock_guard<std::mutex> lock(s->kernelDispatchMutex);
        auto rr_it = s->kernelIdToName.find(kernel_id);
        if (rr_it == s->kernelIdToName.end())
            return;
        count = s->kernelDispatchCount[rr_it->second]++;
    }
    *config = configs[count % configs.size()];
}


void RocprofDataSource::counter_buffer_callback(
    rocprofiler_context_id_t context,
    rocprofiler_buffer_id_t buffer_id,
    rocprofiler_record_header_t** headers,
    size_t num_headers,
    void* user_data,
    uint64_t drop_count)
{
    assert(drop_count == 0 && "drop count should be zero for lossless policy");
    static thread_local rpdtracer::LocalStringCache t_stringCache;

    Logger &logger = Logger::singleton();

    const timestamp_t cb_begin_time = clocktime_ns();
    size_t counter_records = 0;
    size_t dispatch_count = 0;

    // Buffer layout: records arrive as DISPATCH_HEADER followed by its
    // COUNTER_RECORD_VALUE entries, then the next DISPATCH_HEADER, etc.
    // Process streaming: accumulate values for current dispatch, flush
    // when a new header arrives or the buffer ends.

    struct CounterAccum {
        double sum {0.0};
        uint64_t count {0};
    };
    // Per-counter accumulator for the current dispatch (indexed by counter_id.handle)
    constexpr size_t ACCUM_BUCKETS = 16;
    std::array<std::pair<uint64_t, CounterAccum>, ACCUM_BUCKETS> accum_slots{};
    size_t accum_used = 0;
    sqlite3_int64 cur_op_id = 0;
    bool have_dispatch = false;

    auto flush_dispatch = [&]() {
        if (!have_dispatch)
            return;
        for (size_t j = 0; j < accum_used; ++j) {
            auto& [counter_handle, accum] = accum_slots[j];
            auto name_it = s->counterIdNames.find(counter_handle);
            if (name_it == s->counterIdNames.end())
                continue;

            const std::string& counterName = name_it->second;
            bool shouldAverage = s->averagedCounters().count(counterName) > 0;
            double value = shouldAverage ? (accum.sum / accum.count) : accum.sum;

            sqlite3_int64 name_id = t_stringCache.lookup(counterName, logger.stringTable(), logger.storageGeneration());

            CounterTable::row row;
            row.op_id = cur_op_id;
            row.name_id = name_id;
            row.value = value;
            logger.counterTable().insert(row);
        }
        accum_used = 0;
        have_dispatch = false;
    };

    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];
        if (header->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS)
            continue;

        if (header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER) {
            flush_dispatch();

            auto* record = static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
            auto op_it = s->dispatchOpId.find(record->dispatch_info.dispatch_id);
            if (op_it != s->dispatchOpId.end()) {
                cur_op_id = op_it->second;
                have_dispatch = true;
                s->dispatchOpId.erase(op_it);
                dispatch_count++;
            }
        }
        else if (header->kind == ROCPROFILER_COUNTER_RECORD_VALUE && have_dispatch) {
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
            rocprofiler_counter_id_t counter_id = {.handle = 0};
            rocprofiler_query_record_counter_id(record->id, &counter_id);

            size_t slot = accum_used;
            for (size_t j = 0; j < accum_used; ++j) {
                if (accum_slots[j].first == counter_id.handle) {
                    slot = j;
                    break;
                }
            }
            if (slot == accum_used && accum_used < ACCUM_BUCKETS) {
                accum_slots[accum_used] = {counter_id.handle, {0.0, 0}};
                accum_used++;
            }
            if (slot < ACCUM_BUCKETS) {
                accum_slots[slot].second.sum += record->counter_value;
                accum_slots[slot].second.count++;
            }
            counter_records++;
        }
    }
    flush_dispatch();

    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "records=%ld dispatches=%ld", counter_records, dispatch_count);
    logger.createOverheadRecord(cb_begin_time, cb_end_time, "RocprofDataSource::counter_buffer_callback", buff);
}


void RocprofDataSource::code_object_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            for (auto& buffer : s->client_buffers) {
                auto status = rocprofiler_flush_buffer(buffer);
                if (status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                    (void)status;
            }
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            const char *name = cxx_demangle(data->kernel_name);
            s->kernel_names.emplace(data->kernel_id, name);

            if (s->collectCounters) {
                std::lock_guard<std::mutex> lock(s->kernelDispatchMutex);
                s->kernelIdToName[data->kernel_id] = name;
            }
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
    // If a RocprofilerDataSource instance hasn't been create yet, just pass
    if (s == nullptr)
        return nullptr;

    //RocprofDataSourceShared::singleton();	// CRITICAL: static init

    id->name = "rpd_tracer";
    s->clientId = id;

    // return pointer to configure data
    return &s->cfg;
}


int RocprofDataSource::toolInit(rocprofiler_client_finalize_t finialize_func, void* tool_data)
{
    s->finalizer = finialize_func;

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
    apiList.add("hipGetDevicePropertiesR0600");
    apiList.add("hipGetDeviceCount");
    apiList.add("hipDeviceGetAttribute");
    apiList.add("hipRuntimeGetVersion");
    apiList.add("hipPeekAtLastError");
    apiList.add("hipModuleGetFunction");

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

        // roctx handling moved to RoctxDataSource
#if 1
        // Buffers
        constexpr auto buffer_size_bytes      = 0x40000;
        constexpr auto buffer_watermark_bytes = buffer_size_bytes / 8;

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
                                                     ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                                                     apis.data(),
                                                     apis.size(),
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
                                                     nullptr,
                                                     0,
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                                     nullptr,
                                                     0,
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                                                     nullptr,
                                                     0,
                                                     buffer);

        rocprofiler_configure_buffer_tracing_service(context,
                                                     ROCPROFILER_BUFFER_TRACING_KFD_QUEUE,
                                                     nullptr,
                                                     0,
                                                      buffer);

        auto client_thread = rocprofiler_callback_thread_t{};
        rocprofiler_create_callback_thread(&client_thread);
        rocprofiler_assign_callback_thread(buffer, client_thread);
#endif

        // Counter collection (own buffer, own callback thread)
        s->collectCounters = (atoi(rpdtracer::getConfig("RPDT_ROCPROF_COLLECT_COUNTERS", "rocprof_collect_counters", "0")) != 0);
        // Register property unconditionally so rlog-config shows it
        const char *userSets = rpdtracer::getConfig("RPDT_ROCPROF_COUNTER_SETS", "rocprof_counter_sets", "");
        if (s->collectCounters && s->counterSets.empty()) {
            if (userSets[0] != '\0') {
                std::string input(userSets);
                size_t setStart = 0;
                while (setStart < input.size()) {
                    size_t setEnd = input.find(';', setStart);
                    if (setEnd == std::string::npos) setEnd = input.size();
                    std::set<std::string> counterSet;
                    size_t pos = setStart;
                    while (pos < setEnd) {
                        size_t comma = input.find(',', pos);
                        if (comma == std::string::npos || comma > setEnd) comma = setEnd;
                        std::string name = input.substr(pos, comma - pos);
                        if (!name.empty()) counterSet.insert(name);
                        pos = comma + 1;
                    }
                    if (!counterSet.empty()) s->counterSets.push_back(std::move(counterSet));
                    setStart = setEnd + 1;
                }
            }
            if (s->counterSets.empty()) {
                s->counterSets.push_back({"VALUInsts", "VALUBusy", "VALUUtilization"});
                s->counterSets.push_back({"SQ_WAVES", "SALUInsts", "MemUnitBusy"});
            }
            rpdtracer::rpdLog("rpd_tracer: counter collection enabled (%ld sets)\n", s->counterSets.size());
        }
        if (s->collectCounters) {
            rocprofiler_configure_buffer_dispatch_counting_service(
                context,
                buffer,
                RocprofDataSource::counter_dispatch_callback,
                nullptr);
        }

        int isValid = 0;
        rocprofiler_context_is_valid(context, &isValid);
        if (isValid == 0) {
            context.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
        //rocprofiler_start_context(context);
        rocprofiler_stop_context(context);
    }

    return 0;
}

void RocprofDataSource::toolFinialize(void* tool_data)
{
    if (s == nullptr)
        return;

    if (s->utilityContext.handle != 0) {
        rocprofiler_stop_context(s->utilityContext);
        s->utilityContext.handle = 0;
    }
    for (auto &context : s->contexts) {
        if (context.handle != 0) {
            rocprofiler_stop_context(context);
            context.handle = 0;
        }
    }
}

} // namespace rpdtracer

namespace {

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

} // anonymous namespace
