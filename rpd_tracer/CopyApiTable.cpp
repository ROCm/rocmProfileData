// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "NetWriterBackend.h"

#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::CopyApiTable;

namespace rpdtracer {

const char *SCHEMA_COPYAPI = "CREATE TEMPORARY TABLE \"temp_rocpd_copyapi\" (\"api_ptr_id\" integer NOT NULL PRIMARY KEY REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"stream\" varchar(18) NOT NULL, \"size\" integer NOT NULL, \"width\" integer NOT NULL, \"height\" integer NOT NULL, \"kind\" integer NOT NULL, \"dst\" varchar(18) NOT NULL, \"src\" varchar(18) NOT NULL, \"dstDevice\" integer NOT NULL, \"srcDevice\" integer NOT NULL, \"sync\" bool NOT NULL, \"pinned\" bool NOT NULL);";


class CopyApiTableWriterBackend : public WriterBackend
{
public:
    CopyApiTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_COPYAPI, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &m_apiInsert, NULL);
        }
    }

    ~CopyApiTableWriterBackend() {
        sqlite3_finalize(m_apiInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<CopyApiTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            CopyApiTable::row &r = rows[i % capacity];

            sqlite3_bind_int64(m_apiInsert, index++, r.api_id + m_idOffset);
            sqlite3_bind_text(m_apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
            if (r.size > 0)
                sqlite3_bind_int(m_apiInsert, index++, r.size);
            else
                sqlite3_bind_text(m_apiInsert, index++, "", -1, SQLITE_STATIC);
            if (r.width > 0)
                sqlite3_bind_int(m_apiInsert, index++, r.width);
            else
                sqlite3_bind_text(m_apiInsert, index++, "", -1, SQLITE_STATIC);
            if (r.height > 0)
                sqlite3_bind_int(m_apiInsert, index++, r.height);
            else
                sqlite3_bind_text(m_apiInsert, index++, "", -1, SQLITE_STATIC);
            sqlite3_bind_int(m_apiInsert, index++, r.kind);
            sqlite3_bind_text(m_apiInsert, index++, r.dst.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(m_apiInsert, index++, r.src.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(m_apiInsert, index++, r.dstDevice);
            sqlite3_bind_int(m_apiInsert, index++, r.srcDevice);
            sqlite3_bind_int(m_apiInsert, index++, r.sync);
            sqlite3_bind_int(m_apiInsert, index++, r.pinned);
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
        ret = sqlite3_exec(m_conn, "insert into rocpd_copyapi select * from temp_rocpd_copyapi", NULL, NULL, NULL);
        rpdLog("rocpd_copyapi: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_copyapi", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_apiInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* CopyApiTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new CopyApiTableWriterBackend(basefile, directWrite);
}

static void serializeCopyApiTableRow(const void *row, ByteBuffer &buf) {
    static_cast<const CopyApiTable::row*>(row)->serialize(buf);
}

WriterBackend* CopyApiTable::createNetWriterBackend(const char *host, int port, bool directWrite)
{
    return new NetWriterBackend("CopyApiTable", host, port, directWrite,
        sizeof(CopyApiTable::row), serializeCopyApiTableRow);
}


class CopyApiTablePrivate
{
public:
    CopyApiTablePrivate(CopyApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<CopyApiTable::row, BUFFERSIZE> rows; // Circular buffer

    CopyApiTable *p;
};


CopyApiTable::CopyApiTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, CopyApiTablePrivate::BUFFERSIZE, CopyApiTablePrivate::BATCHSIZE,
    isRemoteNode() ? createNetWriterBackend(getLogaggHost(), getLogaggPort(), directWrite)
                   : createWriterBackend(basefile, directWrite))
, d(new CopyApiTablePrivate(this))
{
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
    m_writerBackend->flush();
}


void CopyApiTable::writeRows()
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
    createOverheadRecord(cb_begin_time, cb_end_time, "CopyApiTable::writeRows", buff);
}


void CopyApiTable::row::serialize(ByteBuffer &buf) const {
    buf.writeString(stream);
    buf.writeInt(size);
    buf.writeInt(width);
    buf.writeInt(height);
    buf.writeString(dst);
    buf.writeString(src);
    buf.writeInt(dstDevice);
    buf.writeInt(srcDevice);
    buf.writeInt(kind);
    buf.writeBool(sync);
    buf.writeBool(pinned);
    buf.writeInt64(api_id);
}

void CopyApiTable::row::deserialize(ByteBuffer &buf) {
    stream = buf.readString();
    size = buf.readInt();
    width = buf.readInt();
    height = buf.readInt();
    dst = buf.readString();
    src = buf.readString();
    dstDevice = buf.readInt();
    srcDevice = buf.readInt();
    kind = buf.readInt();
    sync = buf.readBool();
    pinned = buf.readBool();
    api_id = buf.readInt64();
}

}  // namespace rpdtracer
