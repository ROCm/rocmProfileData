// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"

#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::StackFrameTable;

namespace rpdtracer {

const char *SCHEMA = R"sql(CREATE TEMPORARY TABLE "temp_rocpd_stackframe" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "depth" integer NOT NULL, "name_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);)sql";


class StackFrameTableWriterBackend : public WriterBackend
{
public:
    StackFrameTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_stackframe(api_ptr_id, depth, name_id) values (?,?,?)", -1, &m_insertStatement, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_stackframe(api_ptr_id, depth, name_id) values (?,?,?)", -1, &m_insertStatement, NULL);
        }
    }

    ~StackFrameTableWriterBackend() {
        sqlite3_finalize(m_insertStatement);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<StackFrameTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            StackFrameTable::row &r = rows[i % capacity];

            sqlite3_bind_int64(m_insertStatement, index++, r.api_id + m_idOffset);
            sqlite3_bind_int(m_insertStatement, index++, r.depth);
            sqlite3_bind_int64(m_insertStatement, index++, r.name_id);
            sqlite3_step(m_insertStatement);
            sqlite3_reset(m_insertStatement);
        }

        sqlite3_exec(m_conn, "END TRANSACTION", NULL, NULL, NULL);
    }

    void flush() override {
        if (m_directWrite)
            return;
        int ret = 0;
        ret = sqlite3_exec(m_conn, "begin transaction", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "insert into rocpd_stackframe select * from temp_rocpd_stackframe", NULL, NULL, NULL);
        rpdLog("rocpd_stackframe: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_stackframe", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_insertStatement;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* StackFrameTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new StackFrameTableWriterBackend(basefile, directWrite);
}


class StackFrameTablePrivate
{
public:
    StackFrameTablePrivate(StackFrameTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<StackFrameTable::row, BUFFERSIZE> rows; // Circular buffer

    StackFrameTable *p;
};


StackFrameTable::StackFrameTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, StackFrameTablePrivate::BUFFERSIZE, StackFrameTablePrivate::BATCHSIZE,
    createWriterBackend(basefile, directWrite))
, d(new StackFrameTablePrivate(this))
{
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
    m_writerBackend->flush();
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

    m_writerBackend->writeBatch(d->rows.data(), start, end, BUFFERSIZE);

    lock.lock();
    m_tail = end;
    lock.unlock();

    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "StackFrameTable::writeRows", buff);
}


void StackFrameTable::row::serialize(ByteBuffer &buf) const {
    buf.writeInt64(api_id);
    buf.writeInt(depth);
    buf.writeInt64(name_id);
}

void StackFrameTable::row::deserialize(ByteBuffer &buf) {
    api_id = buf.readInt64();
    depth = buf.readInt();
    name_id = buf.readInt64();
}

}  // namespace rpdtracer
