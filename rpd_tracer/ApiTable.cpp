#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_API = "CREATE TEMPORARY TABLE \"temp_rocpd_api\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"pid\" integer NOT NULL, \"tid\" integer NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"apiName_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"args_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";


class ApiTablePrivate
{
public:
    ApiTablePrivate(ApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 16384;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<ApiTable::row, BUFFERSIZE> rows; // Circular buffer
    int head;
    int tail;
    int count;

    std::map<sqlite3_int64, ApiTable::row> inFlight;

    sqlite3_stmt *apiInsert;
    sqlite3_stmt *apiInsertNoId;

    void writeRows();

    void work();                // work thread
    std::thread *worker;

    ApiTable *p;
};


ApiTable::ApiTable(const char *basefile)
: Table(basefile)
, d(new ApiTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_API, NULL, NULL, NULL);

    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &d->apiInsertNoId, NULL);

    d->head = 0;
    d->tail = 0;

    d->worker = NULL;
}

void ApiTable::insert(const ApiTable::row &row)
{
	if (row.phase == 0) {
       d->inFlight.insert({row.api_id, row});
       return;
    }

    if (d->head - d->tail >= ApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        d->writeRows();
    }

    if (row.phase == 1) {
        //std::map<sqlite3_int64, row>::iterator it = d->inFlight.find(row.api_id);
        if (d->inFlight.count(row.api_id) > 0) {
            ApiTable::row &r = d->inFlight[row.api_id];
            r.end = row.end;
            d->rows[++d->head] = r;
            d->inFlight.erase(row.api_id);
            //printf("wrote: %lld\n", row.api_id);
        }
    }

    if ((d->head - d->tail) >= ApiTablePrivate::BATCHSIZE) {
        d->writeRows();
    }
}

void ApiTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void ApiTable::finalize()
{
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_api select * from temp_rocpd_api", NULL, NULL, NULL);
    printf("rocpd_api: %d\n", ret);
}


void ApiTablePrivate::writeRows()
{
    int i = 0;

    if (head == tail)
        return;

    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    while (i < BATCHSIZE && (head > tail + i)) {
        // insert rocpd_api
        int index = 1;
        ApiTable::row &r = rows[tail + i];
        sqlite3_bind_int(apiInsert, index++, r.api_id);
        sqlite3_bind_int(apiInsert, index++, r.pid);
        sqlite3_bind_int(apiInsert, index++, r.tid);
        sqlite3_bind_int64(apiInsert, index++, r.start);
        sqlite3_bind_int64(apiInsert, index++, r.end);
        sqlite3_bind_int64(apiInsert, index++, r.apiName_id);
        sqlite3_bind_int64(apiInsert, index++, r.args_id);
        int ret = sqlite3_step(apiInsert);
        sqlite3_reset(apiInsert);
        ++i;
    }
    tail = tail + i;

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
}


void ApiTablePrivate::work()
{

}


