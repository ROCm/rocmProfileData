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

//Schema
char *SCHEMA_STRING = "CREATE TEMPORARY TABLE \"temp_rocpd_string\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"string\" varchar(4096) NOT NULL)";

char *SCHEMA_API = "CREATE TEMPORARY TABLE \"temp_rocpd_api\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"pid\" integer NOT NULL, \"tid\" integer NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"apiName_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"args_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";

char *SCHEMA_OP = "CREATE TEMPORARY TABLE \"temp_rocpd_op\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"gpuId\" integer NOT NULL, \"queueId\" integer NOT NULL, \"sequenceId\" integer NOT NULL, \"completionSignal\" varchar(18) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"description_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"opType_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";

char *SCHEMA_API_OPS = "CREATE TEMPORARY TABLE \"temp_rocpd_api_ops\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"api_id\" integer NOT NULL REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"op_id\" integer NOT NULL REFERENCES \"rocpd_op\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";

sqlite3 *connection = NULL;
sqlite3_stmt *opInsert = NULL;
sqlite3_stmt *apiInsert = NULL;
sqlite3_stmt *apiInsertNoId = NULL;
sqlite3_stmt *apiOpInsert = NULL;
sqlite3_stmt *stringInsert = NULL;

const sqlite_int64 EMPTY_STRING_ID= 1;


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
        switch (cid) {
            case HIP_API_ID_hipMalloc:
                //entry->ptr = *(data->args.hipMalloc.ptr);
                break;
            case HIP_API_ID_hipModuleLaunchKernel:
                if (data->phase == 0) {
                    const hipFunction_t f = data->args.hipModuleLaunchKernel.f;
                    if (f != NULL) {
                        //printf("t_args=%d, id=%lu, kernel=%s\n", cid, data->correlation_id, cxx_demangle(hipKernelNameRef(f)));
                    } }
                break;
            default:
                break;
        }
    }
    if (domain == ACTIVITY_DOMAIN_ROCTX) {
        const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
        const timestamp_t time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

        uint32_t pid = GetPid();
        uint32_t tid = GetTid();
        printf("t_roctx=%d, message=%s, ts=%lu, pid=%d, tid=%d\n", cid, data->args.message, time, pid, tid);
    }
}

int count = 0;

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

void hip_activity_callback(const char* begin, const char* end, void* arg)
{
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
#if 1
        else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
            const char *name = roctracer_op_string(record->domain, record->op, record->kind);
            int index = 0;
            sqlite_int64 rowId = 0;
            int ret = 0;

            //"insert into rocpd_string(string) values (?)"
            sqlite3_bind_text(stringInsert, 1, name, -1, SQLITE_STATIC);
            ret = sqlite3_step(stringInsert);
            sqlite3_reset(stringInsert);
            rowId = sqlite3_last_insert_rowid(connection);

            // "insert into rocpd_op(gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)"
            index = 1;
            sqlite3_bind_int(opInsert, index++, record->device_id); // gpu
            sqlite3_bind_int(opInsert, index++, record->queue_id);  // queue
            sqlite3_bind_int(opInsert, index++, 0);                 // sequence
            sqlite3_bind_text(opInsert, index++, "", -1, SQLITE_STATIC); // completion signal
            sqlite3_bind_int64(opInsert, index++, record->begin_ns); // start
            sqlite3_bind_int64(opInsert, index++, record->end_ns); // end
            sqlite3_bind_int64(opInsert, index++, EMPTY_STRING_ID); // desc_id
            sqlite3_bind_int64(opInsert, index++, rowId); // op_id

            ret = sqlite3_step(opInsert);
            sqlite3_reset(opInsert);

            rowId = sqlite3_last_insert_rowid(connection);
            index = 1;
            sqlite3_bind_int64(apiOpInsert, index++, record->correlation_id);
            sqlite3_bind_int64(apiOpInsert, index++, rowId);
            ret = sqlite3_step(apiOpInsert);
            sqlite3_reset(apiOpInsert);
        }
#endif
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


void hcc_activity_callback(const char* begin, const char* end, void* arg)
{
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);

    int batchSize = 0;

    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    while (record < end_record) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        int index = 0;
        sqlite_int64 rowId = 0;
        int ret = 0;

            //"insert into rocpd_string(string) values (?)"
            //sqlite3_bind_text(stringInsert, 1, name, -1, SQLITE_STATIC);
            //ret = sqlite3_step(stringInsert);
            //sqlite3_reset(stringInsert);
            //rowId = sqlite3_last_insert_rowid(connection);

            // "insert into rocpd_op(gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)"
            index = 1;
            sqlite3_bind_int(opInsert, index++, record->device_id);	// gpu
            sqlite3_bind_int(opInsert, index++, record->queue_id);  // queue
            sqlite3_bind_int(opInsert, index++, 0);                 // sequence
            sqlite3_bind_text(opInsert, index++, "", -1, SQLITE_STATIC); // completion signal
            sqlite3_bind_int64(opInsert, index++, record->begin_ns); // start
            sqlite3_bind_int64(opInsert, index++, record->end_ns); // end
            sqlite3_bind_int64(opInsert, index++, EMPTY_STRING_ID); // desc_id
            //sqlite3_bind_int64(opInsert, index++, rowId); // op_id
            sqlite3_bind_int64(apiInsert, index++, EMPTY_STRING_ID);

            ret = sqlite3_step(opInsert);
            sqlite3_reset(opInsert);

            rowId = sqlite3_last_insert_rowid(connection);
            index = 1;
            sqlite3_bind_int64(apiOpInsert, index++, record->correlation_id);
            sqlite3_bind_int64(apiOpInsert, index++, rowId);
            ret = sqlite3_step(apiOpInsert);
            sqlite3_reset(apiOpInsert);

            //printf("t_op=%s, id=%lu, begin=%lu, end=%lu, gpuId=%d, queueId=%lu\n", name, record->correlation_id, record->begin_ns, record->end_ns, record->device_id, record->queue_id);
//*/
        roctracer_next_record(record, &record);
        ++batchSize;
    }
    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("### activity_callback hcc ### tid=%d ### %d (%d) %lu \n", GetTid(), count++, batchSize, (cb_end_time - cb_begin_time)/1000);

    // Make a tracer overhead record
    sqlite3_exec(connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);
    create_overhead_record("overhead (hcc)", cb_begin_time, cb_end_time);
    create_overhead_record("prepare", cb_begin_time, cb_mid_time);
    create_overhead_record("commit", cb_mid_time, cb_end_time);
    sqlite3_exec(connection, "END TRANSACTION", NULL, NULL, NULL);
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

#if 1
    // Log hip
    roctracer_properties_t hip_cb_properties;
    memset(&hip_cb_properties, 0, sizeof(roctracer_properties_t));
    hip_cb_properties.buffer_size = 0xf0000;
    hip_cb_properties.buffer_callback_fun = hip_activity_callback;
    //roctracer_pool_t *hipPool;
    roctracer_open_pool_expl(&hip_cb_properties, &hipPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HIP_API, hipPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hipPool);	// FIXME - logging on 1 thread for now
#endif

#if 0
    // Log hcc
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
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

    sqlite3_open("./trace.rpd", &connection);

    //char *api_sql = "CREATE TABLE IF NOT EXISTS 'rocpd_api' ('id' integer NOT NULL PRIMARY KEY AUTOINCREMENT, 'pid' integer NOT NULL, 'tid' integer NOT NULL);";
    //sqlite3_exec(connection, api_sql, NULL, NULL, NULL);

    int ret;

    ret = sqlite3_exec(connection, SCHEMA_STRING, NULL, NULL, NULL);
    printf("create: %d\n", ret);
    ret = sqlite3_exec(connection, SCHEMA_API, NULL, NULL, NULL);
    printf("create: %d\n", ret);
    ret = sqlite3_exec(connection, SCHEMA_OP, NULL, NULL, NULL);
    printf("create: %d\n", ret);
    ret = sqlite3_exec(connection, SCHEMA_API_OPS, NULL, NULL, NULL);
    printf("create: %d\n", ret);

    ret = sqlite3_prepare_v2(connection, "insert into temp_rocpd_op(gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)", -1, &opInsert, NULL);
    ret = sqlite3_prepare_v2(connection, "insert into temp_rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", -1, &apiInsert, NULL); 
    ret = sqlite3_prepare_v2(connection, "insert into temp_rocpd_api_ops(api_id, op_id) values (?,?)", -1, &apiOpInsert, NULL);
    ret = sqlite3_prepare_v2(connection, "insert into temp_rocpd_string(string) values (?)", -1, &stringInsert, NULL);
    ret = sqlite3_prepare_v2(connection, "insert into temp_rocpd_api(pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &apiInsertNoId, NULL);
    printf("ret: %d %d\n", ret, SQLITE_OK);
   
    sqlite3_exec(connection, "insert into temp_rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (4000000000, 0, 0, 0, 0, 0, 0)", NULL, NULL, NULL);
    sqlite3_exec(connection, "delete from temp_rocpd_api where id=4000000000", NULL, NULL, NULL);
    printf("ret: %d %d\n", ret, SQLITE_OK);

    //printf("rpdInit()\n");
    init_tracing();
    start_tracing();
}

void rpdFinalize()
{
    //printf("rpdFinalize\n");
    stop_tracing();

    int ret;

    const timestamp_t begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    ret = sqlite3_exec(connection, "insert into rocpd_api select * from temp_rocpd_api", NULL, NULL, NULL);
    printf("rocpd_api: %d\n", ret);
    ret = sqlite3_exec(connection, "insert into rocpd_op select * from temp_rocpd_op", NULL, NULL, NULL);
    printf("rocpd_op: %d\n", ret);
    ret = sqlite3_exec(connection, "insert into rocpd_api_ops select * from temp_rocpd_api_ops", NULL, NULL, NULL);
    printf("rocpd_api_ops: %d\n", ret);
    ret = sqlite3_exec(connection, "insert into rocpd_string select * from temp_rocpd_string where id != 1", NULL, NULL, NULL);
    printf("rocpd_string: %d\n", ret);
    const timestamp_t end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);

    sqlite3_close(connection);
}

