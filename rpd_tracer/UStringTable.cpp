// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"

#include <thread>
#include <unordered_map>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::UStringTable;

namespace rpdtracer {

const char *SCHEMA_USTRING = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_ustring" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
)|";


class UStringTableWriterBackend : public WriterBackend
{
public:
    UStringTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_USTRING, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_ustring(id, string) values (?,?)", -1, &m_stringInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_ustring(id, string) values (?,?)", -1, &m_stringInsert, NULL);
        }
    }

    ~UStringTableWriterBackend() {
        sqlite3_finalize(m_stringInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<UStringTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            UStringTable::row &r = rows[i % capacity];
            sqlite3_bind_int64(m_stringInsert, index++, r.string_id + m_idOffset);
            sqlite3_bind_text(m_stringInsert, index++, r.string.c_str(), -1, SQLITE_STATIC);
            sqlite3_step(m_stringInsert);
            sqlite3_reset(m_stringInsert);
        }

        sqlite3_exec(m_conn, "END TRANSACTION", NULL, NULL, NULL);
    }

    void flush() override {
        if (m_directWrite)
            return;
        int ret = 0;
        ret = sqlite3_exec(m_conn, "begin transaction", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "insert into rocpd_ustring select * from temp_rocpd_ustring", NULL, NULL, NULL);
        rpdLog("rocpd_ustring: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_ustring", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_stringInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* UStringTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new UStringTableWriterBackend(basefile, directWrite);
}


class UStringTablePrivate
{
public:
    UStringTablePrivate(UStringTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<UStringTable::row, BUFFERSIZE> rows; // Circular buffer

    void insert(UStringTable::row&);

    UStringTable *p;
};


UStringTable::UStringTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, UStringTablePrivate::BUFFERSIZE, UStringTablePrivate::BATCHSIZE,
    createWriterBackend(basefile, directWrite))
, d(new UStringTablePrivate(this))
{
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
    while (p->m_head - p->m_tail >= UStringTablePrivate::BUFFERSIZE) {
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
    m_writerBackend->flush();
}

void UStringTable::writeRows()
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
#if 0
    // FIXME
    if (done == false) {
        char buff[4096];
        std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
        createOverheadRecord(cb_begin_time, cb_end_time, "UStringTable::writeRows", buff);
    }
#endif
}


void UStringTable::row::serialize(ByteBuffer &buf) const {
    buf.writeString(string);
    buf.writeInt64(string_id);
}

void UStringTable::row::deserialize(ByteBuffer &buf) {
    string = buf.readString();
    string_id = buf.readInt64();
}

}  // namespace rpdtracer
