// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "BufferPool.h"

#include <thread>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::StackFrameTable;

namespace rpdtracer {

const char *SCHEMA = R"sql(CREATE TEMPORARY TABLE "temp_rocpd_stackframe" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "depth" integer NOT NULL, "name_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);)sql";

class StackFrameTablePrivate
{
public:
    StackFrameTablePrivate(StackFrameTable *cls) : p(cls) {
        rows = p->m_slot->rows<StackFrameTable::row>();
    }
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    StackFrameTable::row *rows;

    sqlite3_stmt *insertStatement;
    bool directWrite;

    StackFrameTable *p;
};


StackFrameTable::StackFrameTable(const char *basefile, bool directWrite, BufferPool &pool)
: BufferedTable(basefile, pool.allocate<StackFrameTable::row>(StackFrameTablePrivate::BUFFERSIZE, "StackFrameTable"), StackFrameTablePrivate::BATCHSIZE)
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
    while (m_slot->head() - m_slot->tail() >= StackFrameTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }

    d->rows[(++m_slot->head()) % StackFrameTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_slot->head() - m_slot->tail()) >= StackFrameTablePrivate::BATCHSIZE) {
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
    rpdLog("rocpd_stackframe: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_stackframe", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void StackFrameTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_slot->head() == m_slot->tail())
        return;

    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_slot->tail() + 1;
    int end = m_slot->tail() + BATCHSIZE;
    end = (end > m_slot->head()) ? m_slot->head() : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        StackFrameTable::row &r = d->rows[i % m_slot->capacity()];

        sqlite3_bind_int64(d->insertStatement, index++, r.api_id + m_idOffset);
        sqlite3_bind_int(d->insertStatement, index++, r.depth);
        sqlite3_bind_int64(d->insertStatement, index++, r.name_id);
        int ret = sqlite3_step(d->insertStatement);
        sqlite3_reset(d->insertStatement);
    }
    lock.lock();
    m_slot->tail() = end;
    lock.unlock();

    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    // FIXME
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_slot->head() - m_slot->tail());
    createOverheadRecord(cb_begin_time, cb_end_time, "StackFrameTable::writeRows", buff);
}

}  // namespace rpdtracer
