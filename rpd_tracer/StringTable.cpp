// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "NetWriterBackend.h"

#include <thread>
#include <unordered_map>
#include <array>
#include <mutex>
#include <shared_mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::StringTable;

namespace rpdtracer {

const char *SCHEMA_STRING = "CREATE TEMPORARY TABLE \"temp_rocpd_string\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"string\" varchar(4096) NOT NULL)";


class StringTableWriterBackend : public WriterBackend
{
public:
    StringTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_STRING, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_string(id, string) values (?,?)", -1, &m_stringInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_string(id, string) values (?,?)", -1, &m_stringInsert, NULL);
        }
    }

    ~StringTableWriterBackend() {
        sqlite3_finalize(m_stringInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<StringTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            StringTable::row &r = rows[i % capacity];
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
        ret = sqlite3_exec(m_conn, "insert into rocpd_string select * from temp_rocpd_string", NULL, NULL, NULL);
        rpdLog("rocpd_string: %d\n", ret);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_string", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_stringInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* StringTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new StringTableWriterBackend(basefile, directWrite);
}

static void serializeStringTableRow(const void *row, ByteBuffer &buf) {
    static_cast<const StringTable::row*>(row)->serialize(buf);
}

WriterBackend* StringTable::createNetWriterBackend(const char *host, int port, bool directWrite)
{
    return new NetWriterBackend("StringTable", host, port, directWrite,
        sizeof(StringTable::row), serializeStringTableRow);
}


class StringTablePrivate
{
public:
    StringTablePrivate(StringTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<StringTable::row, BUFFERSIZE> rows; // Circular buffer
    std::unordered_map<std::string,sqlite3_int64> cache;     // Cache for string lookups

    void insert(StringTable::row&);

    std::shared_mutex cacheMutex;

    StringTable *p;
};


StringTable::StringTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, StringTablePrivate::BUFFERSIZE, StringTablePrivate::BATCHSIZE,
    isRemoteNode() ? createNetWriterBackend(getLogaggHost(), getLogaggPort(), directWrite)
                   : createWriterBackend(basefile, directWrite))
, d(new StringTablePrivate(this))
{
    d->cache.reserve(64 * 1024);  // Avoid/delay rehashing for typical runs

    StringTable::getOrCreate("");    // empty string is id=1
}

StringTable::~StringTable()
{
    delete d;
}


sqlite3_int64 StringTable::getOrCreate(const std::string &key)
{
    {
        std::shared_lock<std::shared_mutex> guard(d->cacheMutex);
        auto it = d->cache.find(key);
        if (it != d->cache.end())
            return it->second;
    }
    std::unique_lock<std::shared_mutex> guard(d->cacheMutex);
    auto it = d->cache.find(key);
    if (it != d->cache.end())
        return it->second;
    StringTable::row row;
    row.string_id = 0;
    row.string = key;
    d->insert(row);
    d->cache.insert({row.string, row.string_id});
    return row.string_id;
}

void StringTablePrivate::insert(StringTable::row &row)
{
    std::unique_lock<std::mutex> lock(p->m_mutex);
    while (p->m_head - p->m_tail >= StringTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        //const timestamp_t start = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t start = clocktime_ns();
        p->m_wait.notify_one();  // make sure working is running
        p->m_wait.wait(lock);
        //const timestamp_t end = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        //createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::StringTable::insert");
        lock.lock();
    }

    row.string_id = ++(p->m_head);
    rows[p->m_head % StringTablePrivate::BUFFERSIZE] = row;

    if (p->workerRunning() == false && (p->m_head - p->m_tail) >= StringTablePrivate::BATCHSIZE) {
        //lock.unlock();	// FIXME: okay to comment out?
        p->m_wait.notify_one();
    }
}

void StringTable::flushRows()
{
    m_writerBackend->flush();
}

void StringTable::writeRows()
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
        createOverheadRecord(cb_begin_time, cb_end_time, "StringTable::writeRows", buff);
    }
#endif
}


void StringTable::row::serialize(ByteBuffer &buf) const {
    buf.writeString(string);
    buf.writeInt64(string_id);
}

void StringTable::row::deserialize(ByteBuffer &buf) {
    string = buf.readString();
    string_id = buf.readInt64();
}

}  // namespace rpdtracer
