/*********************************************************************************
* Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "NetWriterBackend.h"

#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::CounterTable;

namespace rpdtracer {

const char *SCHEMA_COUNTER = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_counter" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "op_id" bigint NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "name_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "value" real NOT NULL);
)|";


class CounterTableWriterBackend : public WriterBackend
{
public:
    CounterTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_COUNTER, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_counter(op_id, name_id, value) values (?,?,?)", -1, &m_insertStatement, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_counter(op_id, name_id, value) values (?,?,?)", -1, &m_insertStatement, NULL);
        }
    }

    ~CounterTableWriterBackend() {
        sqlite3_finalize(m_insertStatement);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<CounterTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            CounterTable::row &r = rows[i % capacity];

            sqlite3_bind_int64(m_insertStatement, index++, r.op_id + m_idOffset);
            sqlite3_bind_int64(m_insertStatement, index++, r.name_id + m_idOffset);
            sqlite3_bind_double(m_insertStatement, index++, r.value);
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
        ret = sqlite3_exec(m_conn, "insert into rocpd_counter(op_id, name_id, value) select op_id, name_id, value from temp_rocpd_counter", NULL, NULL, NULL);
        rpdLog("rocpd_counter: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_counter", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_insertStatement;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* CounterTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new CounterTableWriterBackend(basefile, directWrite);
}

static void serializeCounterTableRow(const void *row, ByteBuffer &buf) {
    static_cast<const CounterTable::row*>(row)->serialize(buf);
}

WriterBackend* CounterTable::createNetWriterBackend(const char *host, int port, bool directWrite)
{
    return new NetWriterBackend("CounterTable", host, port, directWrite,
        sizeof(CounterTable::row), serializeCounterTableRow);
}


class CounterTablePrivate
{
public:
    CounterTablePrivate(CounterTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<CounterTable::row, BUFFERSIZE> rows; // Circular buffer

    CounterTable *p;
};


CounterTable::CounterTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, CounterTablePrivate::BUFFERSIZE, CounterTablePrivate::BATCHSIZE,
    isRemoteNode() ? createNetWriterBackend(getLogaggHost(), getLogaggPort(), directWrite)
                   : createWriterBackend(basefile, directWrite))
, d(new CounterTablePrivate(this))
{
}


CounterTable::~CounterTable()
{
    delete d;
}


void CounterTable::insert(const CounterTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= CounterTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % CounterTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= CounterTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}


void CounterTable::flushRows()
{
    m_writerBackend->flush();
}


void CounterTable::writeRows()
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
    createOverheadRecord(cb_begin_time, cb_end_time, "CounterTable::writeRows", buff);
}


void CounterTable::row::serialize(ByteBuffer &buf) const {
    buf.writeInt64(op_id);
    buf.writeInt64(name_id);
    buf.writeDouble(value);
}

void CounterTable::row::deserialize(ByteBuffer &buf) {
    op_id = buf.readInt64();
    name_id = buf.readInt64();
    value = buf.readDouble();
}

}  // namespace rpdtracer
