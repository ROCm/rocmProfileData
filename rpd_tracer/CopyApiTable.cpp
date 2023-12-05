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

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_COPYAPI = "CREATE TEMPORARY TABLE \"temp_rocpd_copyapi\" (\"api_ptr_id\" integer NOT NULL PRIMARY KEY REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"stream\" varchar(18) NOT NULL, \"size\" integer NOT NULL, \"width\" integer NOT NULL, \"height\" integer NOT NULL, \"kind\" integer NOT NULL, \"dst\" varchar(18) NOT NULL, \"src\" varchar(18) NOT NULL, \"dstDevice\" integer NOT NULL, \"srcDevice\" integer NOT NULL, \"sync\" bool NOT NULL, \"pinned\" bool NOT NULL);";

class CopyApiTablePrivate
{
public:
    CopyApiTablePrivate(CopyApiTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<CopyApiTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *apiInsert;

    CopyApiTable *p;
};


CopyApiTable::CopyApiTable(const char *basefile)
: BufferedTable(basefile, CopyApiTablePrivate::BUFFERSIZE, CopyApiTablePrivate::BATCHSIZE)
, d(new CopyApiTablePrivate(this))
{
    int ret;
    // set up tmp table
    ret = sqlite3_exec(m_connection, SCHEMA_COPYAPI, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
}


CopyApiTable::~CopyApiTable()
{
    delete d;
}


void CopyApiTable::insert(const CopyApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= CopyApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % CopyApiTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= CopyApiTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}


void CopyApiTable::flushRows()
{
    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_copyapi select * from temp_rocpd_copyapi", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_copyapi: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_copyapi", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void CopyApiTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_head == m_tail)
        return;

    //const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_tail + 1;
    int end = m_tail + BATCHSIZE;
    end = (end > m_head) ? m_head : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        CopyApiTable::row &r = d->rows[i % BUFFERSIZE];

        sqlite3_bind_int64(d->apiInsert, index++, r.api_id + m_idOffset);
        sqlite3_bind_text(d->apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
        if (r.size > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.size);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.width > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.width);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.height > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.height);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        //sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int(d->apiInsert, index++, r.kind);
        sqlite3_bind_text(d->apiInsert, index++, r.dst.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(d->apiInsert, index++, r.src.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(d->apiInsert, index++, r.dstDevice);
        sqlite3_bind_int(d->apiInsert, index++, r.srcDevice);
        sqlite3_bind_int(d->apiInsert, index++, r.sync);
        sqlite3_bind_int(d->apiInsert, index++, r.pinned);
        int ret = sqlite3_step(d->apiInsert);
        sqlite3_reset(d->apiInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "CopyApiTable::writeRows", buff);
}
