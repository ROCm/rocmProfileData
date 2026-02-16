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
#include <unordered_map>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_USTRING = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_ustring" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
)|";

class UStringTablePrivate
{
public:
    UStringTablePrivate(UStringTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<UStringTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *stringInsert;

    void insert(UStringTable::row&);

    UStringTable *p;
};


UStringTable::UStringTable(const char *basefile)
: BufferedTable(basefile, UStringTablePrivate::BUFFERSIZE, UStringTablePrivate::BATCHSIZE)
, d(new UStringTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_USTRING, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_ustring(id, string) values (?,?)", -1, &d->stringInsert, NULL);
    
    // empty string is id=1 - insert it first, now
    UStringTable::row row;
    row.string_id = 0;
    row.string = "";
    d->insert(row);
}

UStringTable::~UStringTable()
{
    delete d;
}


sqlite3_int64 UStringTable::create(const std::string &key)
{
    // dedupe empty strings
    if (key == "")
        return 1;

    // new string, create a row
    UStringTable::row row;
    row.string_id = 0;
    row.string = key;
    d->insert(row);		// string_id gets updated with id
    return row.string_id;
}

void UStringTablePrivate::insert(UStringTable::row &row)
{
    std::unique_lock<std::mutex> lock(p->m_mutex);
    if (p->m_head - p->m_tail >= UStringTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        const timestamp_t start = clocktime_ns();
        p->m_wait.notify_one();  // make sure working is running
        p->m_wait.wait(lock);
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::UStringTable::insert");
        lock.lock();
    }

    row.string_id = ++(p->m_head);
    rows[p->m_head % UStringTablePrivate::BUFFERSIZE] = row;

    if (p->workerRunning() == false && (p->m_head - p->m_tail) >= UStringTablePrivate::BATCHSIZE) {
        //lock.unlock();	// FIXME: okay to comment out?
        p->m_wait.notify_one();
    }
}

void UStringTable::flushRows()
{
    int ret = 0;

    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_ustring select * from temp_rocpd_ustring", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_ustring: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_ustring", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);

}

void UStringTable::writeRows()
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
        // insert rocpd_string
        int index = 1;
        UStringTable::row &r = d->rows[i % BUFFERSIZE];
        //printf("%lld %s\n", r.string_id, r.string.c_str());
        sqlite3_bind_int64(d->stringInsert, index++, r.string_id + m_idOffset);
        sqlite3_bind_text(d->stringInsert, index++, r.string.c_str(), -1, SQLITE_STATIC);	// FIXME SQLITE_TRANSIENT?
        int ret = sqlite3_step(d->stringInsert);
        sqlite3_reset(d->stringInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_end_time = clocktime_ns();
#if 0
    // FIXME
    if (done == false) {
        char buff[4096];
        std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
        createOverheadRecord(cb_begin_time, cb_end_time, "UStringTable::writeRows", buff);
    }
#endif
}
