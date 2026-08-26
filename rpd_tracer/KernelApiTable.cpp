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

using rpdtracer::KernelApiTable;

namespace rpdtracer {

const char *SCHEMA_KERNELAPI = R"|(
CREATE TEMPORARY TABLE  "temp_rocpd_kernelapi" ("api_ptr_id" bigint NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "gridX" integer NOT NULL, "gridY" integer NOT NULL, "gridZ" integer NOT NULL, "workgroupX" integer NOT NULL, "workgroupY" integer NOT NULL, "workgroupZ" integer NOT NULL, "groupSegmentSize" integer NOT NULL, "privateSegmentSize" integer NOT NULL, "kernelName_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED)
)|";


class KernelApiTableWriterBackend : public WriterBackend
{
public:
    KernelApiTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_KERNELAPI, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, kernelName_id) values (?,?,?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, kernelName_id) values (?,?,?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        }
    }

    ~KernelApiTableWriterBackend() {
        sqlite3_finalize(m_apiInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<KernelApiTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            KernelApiTable::row &r = rows[i % capacity];
            sqlite3_bind_int64(m_apiInsert, index++, r.api_id + m_idOffset);
            sqlite3_bind_text(m_apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(m_apiInsert, index++, r.gridX);
            sqlite3_bind_int(m_apiInsert, index++, r.gridY);
            sqlite3_bind_int(m_apiInsert, index++, r.gridZ);
            sqlite3_bind_int(m_apiInsert, index++, r.workgroupX);
            sqlite3_bind_int(m_apiInsert, index++, r.workgroupY);
            sqlite3_bind_int(m_apiInsert, index++, r.workgroupZ);
            sqlite3_bind_int(m_apiInsert, index++, r.groupSegmentSize);
            sqlite3_bind_int(m_apiInsert, index++, r.privateSegmentSize);
            sqlite3_bind_int64(m_apiInsert, index++, r.kernelName_id + m_idOffset);
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
        ret = sqlite3_exec(m_conn, "insert into rocpd_kernelapi select * from temp_rocpd_kernelapi", NULL, NULL, NULL);
        rpdLog("rocpd_kernelapi: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_kernelapi", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_apiInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* KernelApiTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new KernelApiTableWriterBackend(basefile, directWrite);
}


class KernelApiTablePrivate
{
public:
    KernelApiTablePrivate(KernelApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<KernelApiTable::row, BUFFERSIZE> rows; // Circular buffer

    KernelApiTable *p;
};


KernelApiTable::KernelApiTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, KernelApiTablePrivate::BUFFERSIZE, KernelApiTablePrivate::BATCHSIZE,
    createWriterBackend(basefile, directWrite))
, d(new KernelApiTablePrivate(this))
{
}


KernelApiTable::~KernelApiTable()
{
    delete d;
}


void KernelApiTable::insert(const KernelApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= KernelApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % KernelApiTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= KernelApiTablePrivate::BATCHSIZE) {
        //lock.unlock();
        m_wait.notify_one();
    }
}


void KernelApiTable::flushRows()
{
    m_writerBackend->flush();
}


void KernelApiTable::writeRows()
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
    createOverheadRecord(cb_begin_time, cb_end_time, "KernelApiTable::writeRows", buff);
}


void KernelApiTable::row::serialize(ByteBuffer &buf) const {
    buf.writeString(stream);
    buf.writeInt(gridX);
    buf.writeInt(gridY);
    buf.writeInt(gridZ);
    buf.writeInt(workgroupX);
    buf.writeInt(workgroupY);
    buf.writeInt(workgroupZ);
    buf.writeInt(groupSegmentSize);
    buf.writeInt(privateSegmentSize);
    buf.writeInt64(kernelName_id);
    buf.writeInt64(api_id);
}

void KernelApiTable::row::deserialize(ByteBuffer &buf) {
    stream = buf.readString();
    gridX = buf.readInt();
    gridY = buf.readInt();
    gridZ = buf.readInt();
    workgroupX = buf.readInt();
    workgroupY = buf.readInt();
    workgroupZ = buf.readInt();
    groupSegmentSize = buf.readInt();
    privateSegmentSize = buf.readInt();
    kernelName_id = buf.readInt64();
    api_id = buf.readInt64();
}

}  // namespace rpdtracer
