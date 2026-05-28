// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "NetWriterBackend.h"

#include <thread>
#include <array>
#include <mutex>

#include "Utility.h"

using rpdtracer::ApiTable;

namespace rpdtracer {

const char *SCHEMA_API = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_api" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "category_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "domain_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" bigint NOT NULL REFERENCES "rocpd_ustring" ("id") DEFERRABLE INITIALLY DEFERRED)
)|";


class ApiTableWriterBackend : public WriterBackend
{
public:
    ApiTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_API, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_api(id, pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_api(id, pid, tid, start, end, domain_id, category_id, apiName_id, args_id) values (?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        }
    }

    ~ApiTableWriterBackend() {
        sqlite3_finalize(m_apiInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<ApiTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            ApiTable::row &r = rows[i % capacity];
            sqlite3_bind_int64(m_apiInsert, index++, r.api_id + m_idOffset);
            sqlite3_bind_int(m_apiInsert, index++, r.pid);
            sqlite3_bind_int(m_apiInsert, index++, r.tid);
            sqlite3_bind_int64(m_apiInsert, index++, r.start);
            sqlite3_bind_int64(m_apiInsert, index++, r.end);
            sqlite3_bind_int64(m_apiInsert, index++, r.domain_id + m_idOffset);
            sqlite3_bind_int64(m_apiInsert, index++, r.category_id + m_idOffset);
            sqlite3_bind_int64(m_apiInsert, index++, r.apiName_id + m_idOffset);
            sqlite3_bind_int64(m_apiInsert, index++, r.args_id + m_idOffset);
            sqlite3_step(m_apiInsert);
            sqlite3_reset(m_apiInsert);
        }

        sqlite3_exec(m_conn, "END TRANSACTION", NULL, NULL, NULL);
    }

    void flush() override {
        if (m_directWrite)
            return;
        int ret = 0;
        ret = sqlite3_exec(m_conn, "begin transaction", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "insert into rocpd_api select * from temp_rocpd_api", NULL, NULL, NULL);
        rpdLog("rocpd_api: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_api", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_apiInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* ApiTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new ApiTableWriterBackend(basefile, directWrite);
}

static void serializeApiTableRow(const void *row, ByteBuffer &buf) {
    static_cast<const ApiTable::row*>(row)->serialize(buf);
}

WriterBackend* ApiTable::createNetWriterBackend(const char *host, int port, bool directWrite)
{
    return new NetWriterBackend("ApiTable", host, port, directWrite,
        sizeof(ApiTable::row), serializeApiTableRow);
}


class ApiTablePrivate
{
public:
    ApiTablePrivate(ApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 16;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<ApiTable::row, BUFFERSIZE> rows; // Circular buffer

    ApiTable *p;
};


ApiTable::ApiTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, ApiTablePrivate::BUFFERSIZE, ApiTablePrivate::BATCHSIZE,
    isRemoteNode() ? createNetWriterBackend(getLogaggHost(), getLogaggPort(), directWrite)
                   : createWriterBackend(basefile, directWrite))
, d(new ApiTablePrivate(this))
{
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
    m_writerBackend->flush();
}


void ApiTable::writeRows()
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
    createOverheadRecord(cb_begin_time, cb_end_time, "ApiTable::writeRows", buff);
}


void ApiTable::row::serialize(ByteBuffer &buf) const {
    buf.writeInt(pid);
    buf.writeInt(tid);
    buf.writeInt64(start);
    buf.writeInt64(end);
    buf.writeInt64(domain_id);
    buf.writeInt64(category_id);
    buf.writeInt64(apiName_id);
    buf.writeInt64(args_id);
    buf.writeInt64(api_id);
}

void ApiTable::row::deserialize(ByteBuffer &buf) {
    pid = buf.readInt();
    tid = buf.readInt();
    start = buf.readInt64();
    end = buf.readInt64();
    domain_id = buf.readInt64();
    category_id = buf.readInt64();
    apiName_id = buf.readInt64();
    args_id = buf.readInt64();
    api_id = buf.readInt64();
}

}  // namespace rpdtracer
