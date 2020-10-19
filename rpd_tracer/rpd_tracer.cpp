#include <atomic>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#include "hip/hip_runtime.h"

#include "hsa_rsrc_factory.h"

#include <sqlite3.h>

#include "Table.h"


static void rpdInit() __attribute__((constructor));
static void rpdFinalize() __attribute__((destructor));

void init_tracing();
void start_tracing();
void stop_tracing();


typedef uint64_t timestamp_t;


#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#include <roctracer_ext.h>
#include <roctracer_roctx.h>
#include <roctx.h>

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#include <cxxabi.h>

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}


#if 0
sqlite3 *connection = NULL;
sqlite3_stmt *apiInsert = NULL;
sqlite3_stmt *apiInsertNoId = NULL;
sqlite3_stmt *stringInsert = NULL;
#endif

const sqlite_int64 EMPTY_STRING_ID = 1;



// Table Recorders
StringTable *s_stringTable = NULL;
OpTable *s_opTable = NULL;
ApiTable *s_apiTable = NULL;



void api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
    //printf("  api_callback\n");
    if (domain == ACTIVITY_DOMAIN_HIP_API) {
        const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
        //printf("ACTIVITY_DOMAIN_HIP_API cid = %d, phase = %d, cor_id = %lu\n", cid, data->phase, data->correlation_id);
        const char *name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
        sqlite3_int64 name_id = s_stringTable->getOrCreate(name);
        char buff[4096];

        ApiTable::row row;
        row.pid = GetPid();
        row.tid = GetTid();
        row.start = 0;
        row.end = 0;
        row.phase = data->phase;
        row.apiName_id = name_id;
        row.args_id = EMPTY_STRING_ID;
        row.api_id = data->correlation_id;

        if (row.phase == 1)  // Log timestamp
            row.end = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
            switch (cid) {
                case HIP_API_ID_hipMalloc:
                    std::snprintf(buff, 4096, "size=0x%x",
                        (uint32_t)(data->args.hipMalloc.size));
                    row.args_id = s_stringTable->getOrCreate(std::string(buff)); 
                    break;
                case HIP_API_ID_hipModuleLaunchKernel:
                    {
                    const hipFunction_t f = data->args.hipModuleLaunchKernel.f;
                    if (f != NULL) {
                        std::string kernelName(cxx_demangle(hipKernelNameRef(f)));
                        std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                            data->args.hipModuleLaunchKernel.stream,
                            kernelName.c_str());
                        row.args_id = s_stringTable->getOrCreate(std::string(buff));
						// Associate kernel name with op
                        sqlite3_int64 kernelName_id = s_stringTable->getOrCreate(kernelName);
                        s_opTable->associateDescription(row.api_id, kernelName_id);
                    } }
                    break;
                case HIP_API_ID_hipMemcpy:
                    std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u", 
                        data->args.hipMemcpy.dst,
                        data->args.hipMemcpy.src,
                        (uint32_t)(data->args.hipMemcpy.sizeBytes),
                        (uint32_t)(data->args.hipMemcpy.kind));
                    row.args_id = s_stringTable->getOrCreate(std::string(buff)); 
                    break;
                default:
                    break;
            }
        }
        else {   // (data->phase == ACTIVITY_API_PHASE_???)

        }
#if 0
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        SPRINT("ptr(%p) size(0x%x)",
          data->args.hipMalloc.ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        SPRINT("ptr(%p)", data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        SPRINT("kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
#endif

        if (row.phase == 0) // Log timestamp
            row.start = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
        
        s_apiTable->insert(row);

    }
    if (domain == ACTIVITY_DOMAIN_ROCTX) {
        const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
        const timestamp_t time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

        uint32_t pid = GetPid();
        uint32_t tid = GetTid();
        //printf("t_roctx=%d, message=%s, ts=%lu, pid=%d, tid=%d\n", cid, data->args.message, time, pid, tid);
    }
}

int count = 0;

// FIXME - we want this
#if 0
void create_overhead_record(char *message, timestamp_t begin, timestamp_t end)
{
    sqlite3_bind_text(stringInsert, 1, message, -1, SQLITE_STATIC);
    int ret = sqlite3_step(stringInsert);
    sqlite3_reset(stringInsert);
    sqlite_int64 rowId = sqlite3_last_insert_rowid(connection);
    //printf("string %ul\n", rowId);

    int index = 1;
    sqlite3_bind_int(apiInsertNoId, index++, GetPid());
    sqlite3_bind_int(apiInsertNoId, index++, GetTid());
    sqlite3_bind_int64(apiInsertNoId, index++, begin);
    sqlite3_bind_int64(apiInsertNoId, index++, end);
    sqlite3_bind_int64(apiInsertNoId, index++, rowId);
    sqlite3_bind_int64(apiInsertNoId, index++, EMPTY_STRING_ID);

    ret = sqlite3_step(apiInsertNoId);
    //printf("  try: %d\n", ret);
    sqlite3_reset(apiInsertNoId);
    //printf("  (%s): %lu    (%lu) \n", message, (end - begin) / 1000, sqlite3_last_insert_rowid(connection));
}
#endif

#if 0
void hip_activity_callback(const char* begin, const char* end, void* arg)
{
return;
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

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
    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("### activity_callback hip ### tid=%d ### %d (%d) %lu \n", GetTid(), count++, batchSize, (cb_end_time - cb_begin_time)/1000);

    // Make a tracer overhead record
    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);
    create_overhead_record("overhead (hip)", cb_begin_time, cb_end_time);
    create_overhead_record("prepare", cb_begin_time, cb_mid_time);
    create_overhead_record("commit", cb_mid_time, cb_end_time);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
}
#endif


void hcc_activity_callback(const char* begin, const char* end, void* arg)
{
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

    int batchSize = 0;

    while (record < end_record) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);

        // FIXME: get_create string_id for 'name' from stringTable
        sqlite3_int64 name_id = s_stringTable->getOrCreate(name);

        OpTable::row row;
        row.gpuId = record->device_id;
        row.queueId = record->queue_id;
        row.sequenceId = 0;
        //row.completionSignal = "";	//strcpy
        strncpy(row.completionSignal, "", 18);
        row.start = record->begin_ns;
        row.end = record->end_ns;
        row.description_id = EMPTY_STRING_ID;
        row.opType_id = name_id;
        row.api_id = record->correlation_id; 

        s_opTable->insert(row);

        roctracer_next_record(record, &record);
        ++batchSize;
    }
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //printf("### activity_callback hcc ### tid=%d ### %d (%d) %lu \n", GetTid(), count++, batchSize, (cb_end_time - cb_begin_time)/1000);

#if 0
    // Make a tracer overhead record
    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);
    create_overhead_record("overhead (hcc)", cb_begin_time, cb_end_time);
    create_overhead_record("prepare", cb_begin_time, cb_mid_time);
    create_overhead_record("commit", cb_mid_time, cb_end_time);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
#endif
}


roctracer_pool_t *hipPool;
roctracer_pool_t *hccPool;

void init_tracing() {
    //printf("# INIT #############################\n");

    // roctracer properties
    //    Whatever the hell that means.  Magic encantation, thanks.
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);

    // Enable API callbacks
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL);

#if 1 
    // Work around a roctracer bug.  Must have a default pool or crash at exit
    // Allocating tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    //properties.buffer_size = 0x40000;
    //properties.buffer_callback_fun = hip_activity_callback;
    roctracer_open_pool(&properties);
    //roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
#endif

#if 0
    // Log hip
    roctracer_properties_t hip_cb_properties;
    memset(&hip_cb_properties, 0, sizeof(roctracer_properties_t));
    //hip_cb_properties.buffer_size = 0xf0000;
    hip_cb_properties.buffer_size = 0x1000;
    hip_cb_properties.buffer_callback_fun = hip_activity_callback;
    //roctracer_pool_t *hipPool;
    roctracer_open_pool_expl(&hip_cb_properties, &hipPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HIP_API, hccPool);
#endif

#if 1
    // Log hcc
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    //hcc_cb_properties.buffer_size = 0x1000; //0x40000;
    hcc_cb_properties.buffer_size = 0x40000;
    hcc_cb_properties.buffer_callback_fun = hcc_activity_callback;
    //roctracer_pool_t *hccPool;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool);
#endif

    //roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_ROCTX);
}

void start_tracing() {
    printf("# START ############################# %d\n", GetTid());
    roctracer_start();
}

void stop_tracing() {
    printf("# STOP #############################\n");
    roctracer_stop();
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS);

    printf("flush -->\n");
    roctracer_flush_activity();
    roctracer_flush_activity_expl(hipPool);
    roctracer_flush_activity_expl(hccPool);
    printf("<--\n");

}


void rpdInit()
{
    printf("rpd_tracer, because\n");

    const char *filename = "./trace.rpd";

    //sqlite3_open("./trace.rpd", &connection);

    // Create table recorders

    s_stringTable = new StringTable(filename);
    s_opTable = new OpTable(filename);
    s_apiTable = new ApiTable(filename);

    printf("rpdInit()\n");
    init_tracing();
    start_tracing();
}

void rpdFinalize()
{
    //printf("rpdFinalize\n");
    stop_tracing();

    // Flush recorders
    const timestamp_t begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    s_stringTable->finalize();
    s_opTable->finalize();
    s_apiTable->finalize();
    const timestamp_t end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);

    //sqlite3_close(connection);
}

