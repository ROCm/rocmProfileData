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
#include "Table.h"

#include <thread>
#include <deque>
#include <map>

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_API = "CREATE TEMPORARY TABLE \"temp_rocpd_api\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"pid\" integer NOT NULL, \"tid\" integer NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"apiName_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"args_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";


class ApiTablePrivate
{
public:
    ApiTablePrivate(ApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 16;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<ApiTable::row, BUFFERSIZE> rows; // Circular buffer

    std::map<std::pair<sqlite3_int64, sqlite3_int64>, std::deque<ApiTable::row>> roctxStacks;

    sqlite3_stmt *apiInsert;
    sqlite3_stmt *apiInsertNoId;

    sqlite3_int64 roctxResumeTime;

    ApiTable *p;
};


ApiTable::ApiTable(const char *basefile)
: BufferedTable(basefile, ApiTablePrivate::BUFFERSIZE, ApiTablePrivate::BATCHSIZE)
, d(new ApiTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_API, NULL, NULL, NULL);

    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &d->apiInsertNoId, NULL);

    d->roctxResumeTime = 0;
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

static sqlite3_int64 roctx_id_hack = sqlite3_int64(1) << 31;

void ApiTable::insertRoctx(ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_head - m_tail >= ApiTablePrivate::BUFFERSIZE) {
        //const timestamp_t start = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t start = clocktime_ns();
        m_wait.notify_one();
        m_wait.wait(lock);
        //const timestamp_t end = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::ApiTable::insert");
        lock.lock();
    }
    row.api_id = ++roctx_id_hack;
    d->rows[(++m_head) % ApiTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= ApiTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}

void ApiTable::pushRoctx(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto key = std::pair<sqlite3_int64, sqlite3_int64>(row.pid, row.tid);
    auto &stack = d->roctxStacks[key];
    stack.push_front(row);
}

void ApiTable::popRoctx(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_head - m_tail >= ApiTablePrivate::BUFFERSIZE) {
        //const timestamp_t start = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t start = clocktime_ns();
        m_wait.notify_one();
        m_wait.wait(lock);
        //const timestamp_t end = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::ApiTable::insert");
        lock.lock();
    }
    auto key = std::pair<sqlite3_int64, sqlite3_int64>(row.pid, row.tid);
    auto &stack = d->roctxStacks[key];
    if (stack.empty() == false) {
        ApiTable::row &r = stack.front();
        r.end = row.end;
        r.api_id = ++roctx_id_hack;
        d->rows[(++m_head) % ApiTablePrivate::BUFFERSIZE] = r;
        stack.pop_front();
    }
    else {  // Pop without a push.  This is due to suspend/resume.  Fudge the start.
        ApiTable::row &r = const_cast<ApiTable::row&>(row);
        r.start = d->roctxResumeTime;
        r.api_id = ++roctx_id_hack;
        d->rows[(++m_head) % ApiTablePrivate::BUFFERSIZE] = r;
    }

    if (workerRunning() == false && (m_head - m_tail) >= ApiTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}

void ApiTable::suspendRoctx(sqlite3_int64 atTime)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_head - m_tail >= ApiTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }

    // Profiling is suspended, we won't get pops for anything pushed.  So end them all now.
    auto it = d->roctxStacks.begin();
    while (it != d->roctxStacks.end()) {
        auto &stack = it->second;
        while (stack.empty() == false) {
            // Make sure there is room
            if (m_head - m_tail >= ApiTablePrivate::BUFFERSIZE) {
                m_wait.notify_one();
                m_wait.wait(lock);
            }
            ApiTable::row &r = stack.front();
            r.end = atTime;
            r.api_id = ++roctx_id_hack;
            d->rows[(++m_head) % ApiTablePrivate::BUFFERSIZE] = r;
            stack.pop_front();
        }
        ++it;
    }

    if (workerRunning() == false && (m_head - m_tail) >= ApiTablePrivate::BATCHSIZE)
        m_wait.notify_one();
}

void ApiTable::resumeRoctx(sqlite3_int64 atTime)
{
    d->roctxResumeTime = atTime;
}


void ApiTable::flushRows()
{
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
