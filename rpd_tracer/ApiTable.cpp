// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"

#include <thread>
#include <array>
#include <mutex>

#include "Utility.h"

using rpdtracer::ApiTable;

namespace rpdtracer {

const char *SCHEMA_API = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_api" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "category_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "domain_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" bigint NOT NULL REFERENCES "rocpd_ustring" ("id") DEFERRABLE INITIALLY DEFERRED)
)|";

class ApiTablePrivate
{
public:
    ApiTablePrivate(ApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 16;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<ApiTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *apiInsert;
    sqlite3_stmt *apiInsertNoId;
    bool directWrite;

    ApiTable *p;
};


ApiTable::ApiTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, ApiTablePrivate::BUFFERSIZE, ApiTablePrivate::BATCHSIZE)
, d(new ApiTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA_API, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(id, pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &d->apiInsertNoId, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_api(id, pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_api(pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &d->apiInsertNoId, NULL);
    }

}


ApiTable::~ApiTable()
{
    delete d;
}


void ApiTable::insert(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_head - m_tail >= ApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        //const timestamp_t start = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t start = clocktime_ns();
        // FIXME: overhead record here
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
        //const timestamp_t end = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::ApiTable::insert");
        lock.lock();
    }

    d->rows[(++m_head) % ApiTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= ApiTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}



void ApiTable::flushRows()
{
    if (d->directWrite)
        return;

    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_api select * from temp_rocpd_api", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_api: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_api", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void ApiTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_head == m_tail)
        return;

    //const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_tail + 1;
    int end = m_tail + BATCHSIZE;
    end = (end > m_head) ? m_head : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        // insert rocpd_api
        int index = 1;
        ApiTable::row &r = d->rows[i % BUFFERSIZE];
        sqlite3_bind_int64(d->apiInsert, index++, r.api_id + m_idOffset);
        sqlite3_bind_int(d->apiInsert, index++, r.pid);
        sqlite3_bind_int(d->apiInsert, index++, r.tid);
        sqlite3_bind_int64(d->apiInsert, index++, r.start);
        sqlite3_bind_int64(d->apiInsert, index++, r.end);
        sqlite3_bind_int64(d->apiInsert, index++, r.domain_id + m_idOffset);
        sqlite3_bind_int64(d->apiInsert, index++, r.category_id + m_idOffset);
        sqlite3_bind_int64(d->apiInsert, index++, r.apiName_id + m_idOffset);
        sqlite3_bind_int64(d->apiInsert, index++, r.args_id + m_idOffset);
        int ret = sqlite3_step(d->apiInsert);
        sqlite3_reset(d->apiInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "ApiTable::writeRows", buff);
}

}  // namespace rpdtracer
