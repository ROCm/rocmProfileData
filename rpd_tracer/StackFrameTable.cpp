// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"

#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::StackFrameTable;

namespace rpdtracer {

const char *SCHEMA = R"sql(CREATE TEMPORARY TABLE "temp_rocpd_stackframe" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "depth" integer NOT NULL, "name_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);)sql";

class StackFrameTablePrivate
{
public:
    StackFrameTablePrivate(StackFrameTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<StackFrameTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *insertStatement;
    bool directWrite;

    StackFrameTable *p;
};


StackFrameTable::StackFrameTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, StackFrameTablePrivate::BUFFERSIZE, StackFrameTablePrivate::BATCHSIZE)
, d(new StackFrameTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_stackframe(api_ptr_id, depth, name_id) values (?,?,?)", -1, &d->insertStatement, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_stackframe(api_ptr_id, depth, name_id) values (?,?,?)", -1, &d->insertStatement, NULL);
    }
}


StackFrameTable::~StackFrameTable()
{
    delete d;
}


void StackFrameTable::insert(const StackFrameTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= StackFrameTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % StackFrameTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= StackFrameTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}


void StackFrameTable::flushRows()
{
    if (d->directWrite)
        return;

    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_stackframe select * from temp_rocpd_stackframe", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_stackframe: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_stackframe", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void StackFrameTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_head == m_tail)
        return;

    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_tail + 1;
    int end = m_tail + BATCHSIZE;
    end = (end > m_head) ? m_head : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        StackFrameTable::row &r = d->rows[i % BUFFERSIZE];

        sqlite3_bind_int64(d->insertStatement, index++, r.api_id + m_idOffset);
        sqlite3_bind_int(d->insertStatement, index++, r.depth);
        sqlite3_bind_int64(d->insertStatement, index++, r.name_id);
        int ret = sqlite3_step(d->insertStatement);
        sqlite3_reset(d->insertStatement);
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
    createOverheadRecord(cb_begin_time, cb_end_time, "StackFrameTable::writeRows", buff);
}

}  // namespace rpdtracer
