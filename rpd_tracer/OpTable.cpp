// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"

#include <map>
#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::OpTable;

namespace rpdtracer {

const char *SCHEMA_OP = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_op" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
)|";

const char *SCHEMA_API_OPS = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" bigint NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" bigint NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
)|";


class OpTableWriterBackend : public WriterBackend
{
public:
    OpTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_OP, NULL, NULL, NULL);
            sqlite3_exec(m_conn, SCHEMA_API_OPS, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_op(id, gpuId, queueId, sequenceId, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)", -1, &m_opInsert, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_api_ops(api_id, op_id) values (?,?)", -1, &m_apiOpInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_op(id, gpuId, queueId, sequenceId, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)", -1, &m_opInsert, NULL);
            sqlite3_prepare_v2(m_conn, "insert into rocpd_api_ops(api_id, op_id) values (?,?)", -1, &m_apiOpInsert, NULL);
        }
    }

    ~OpTableWriterBackend() {
        sqlite3_finalize(m_opInsert);
        sqlite3_finalize(m_apiOpInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<OpTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            OpTable::row &r = rows[i % capacity];
            sqlite3_int64 primaryKey = i + m_idOffset;

            sqlite3_bind_int64(m_opInsert, index++, primaryKey);
            sqlite3_bind_int(m_opInsert, index++, r.gpuId);
            sqlite3_bind_int(m_opInsert, index++, r.queueId);
            sqlite3_bind_int(m_opInsert, index++, r.sequenceId);
            sqlite3_bind_int64(m_opInsert, index++, r.start);
            sqlite3_bind_int64(m_opInsert, index++, r.end);
            sqlite3_bind_int64(m_opInsert, index++, r.description_id + m_idOffset);
            sqlite3_bind_int64(m_opInsert, index++, r.opType_id + m_idOffset);
            sqlite3_step(m_opInsert);
            sqlite3_reset(m_opInsert);

            index = 1;
            sqlite3_bind_int64(m_apiOpInsert, index++, sqlite3_int64(r.api_id) + m_idOffset);
            sqlite3_bind_int64(m_apiOpInsert, index++, sqlite3_int64(i) + m_idOffset);
            sqlite3_step(m_apiOpInsert);
            sqlite3_reset(m_apiOpInsert);
        }

        sqlite3_exec(m_conn, "END TRANSACTION", NULL, NULL, NULL);
    }

    void flush() override {
        if (m_directWrite)
            return;
        int ret = 0;
        ret = sqlite3_exec(m_conn, "begin transaction", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "insert into rocpd_op select * from temp_rocpd_op", NULL, NULL, NULL);
        rpdLog("rocpd_op: %d\n", ret);
        ret = sqlite3_exec(m_conn, "insert into rocpd_api_ops (api_id, op_id) select api_id, op_id from temp_rocpd_api_ops", NULL, NULL, NULL);
        rpdLog("rocpd_api_ops: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_op", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_api_ops", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_opInsert;
    sqlite3_stmt *m_apiOpInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* OpTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new OpTableWriterBackend(basefile, directWrite);
}


class OpTablePrivate
{
public:
    OpTablePrivate(OpTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<OpTable::row, BUFFERSIZE> rows; // Circular buffer

    OpTable *p;
};


OpTable::OpTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, OpTablePrivate::BUFFERSIZE, OpTablePrivate::BATCHSIZE,
    createWriterBackend(basefile, directWrite))
, d(new OpTablePrivate(this))
{
}


OpTable::~OpTable()
{
    delete d;
}


void OpTable::insert(const OpTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= OpTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % OpTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= OpTablePrivate::BATCHSIZE) {
        m_wait.notify_one();
    }
}

void OpTable::associateDescription(const sqlite3_int64 &api_id, const sqlite3_int64 &string_id)
{
#if 0
    std::lock_guard<std::mutex> guard(d->descriptionLock);
    d->descriptions[api_id] = string_id;
#endif
}

void OpTable::flushRows()
{
    m_writerBackend->flush();
}


void OpTable::writeRows()
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

    const timestamp_t cb_end_time = clocktime_ns() + 1;
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "OpTable::writeRows", buff);
}


void OpTable::row::serialize(ByteBuffer &buf) const {
    buf.writeInt(gpuId);
    buf.writeInt(queueId);
    buf.writeInt(sequenceId);
    buf.writeInt64(start);
    buf.writeInt64(end);
    buf.writeInt64(description_id);
    buf.writeInt64(opType_id);
    buf.writeInt64(api_id);
}

void OpTable::row::deserialize(ByteBuffer &buf) {
    gpuId = buf.readInt();
    queueId = buf.readInt();
    sequenceId = buf.readInt();
    start = buf.readInt64();
    end = buf.readInt64();
    description_id = buf.readInt64();
    opType_id = buf.readInt64();
    api_id = buf.readInt64();
}

}  // namespace rpdtracer
