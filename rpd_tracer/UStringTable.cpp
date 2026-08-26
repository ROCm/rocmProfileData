// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "BufferPool.h"

#include <thread>
#include <unordered_map>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::UStringTable;

namespace rpdtracer {

const char *SCHEMA_USTRING = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_ustring" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
)|";

class UStringTablePrivate
{
public:
    UStringTablePrivate(UStringTable *cls) : p(cls) {
        rows = p->m_slot->rows<UStringTable::row>();
    }
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    UStringTable::row *rows;

    sqlite3_stmt *stringInsert;
    bool directWrite;

    void insert(UStringTable::row&);

    UStringTable *p;
};


UStringTable::UStringTable(const char *basefile, bool directWrite, BufferPool &pool)
: BufferedTable(basefile, pool.allocate<UStringTable::row>(UStringTablePrivate::BUFFERSIZE, "UStringTable"), UStringTablePrivate::BATCHSIZE)
, d(new UStringTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA_USTRING, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_ustring(id, string) values (?,?)", -1, &d->stringInsert, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_ustring(id, string) values (?,?)", -1, &d->stringInsert, NULL);
    }

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
    while (p->m_slot->head() - p->m_slot->tail() >= UStringTablePrivate::BUFFERSIZE) {
        const timestamp_t start = clocktime_ns();
        p->m_wait.notify_one();
        p->m_wait.wait(lock);
        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::UStringTable::insert");
        lock.lock();
    }

    row.string_id = ++(p->m_slot->head());
    rows[p->m_slot->head() % UStringTablePrivate::BUFFERSIZE] = row;

    if (p->workerRunning() == false && (p->m_slot->head() - p->m_slot->tail()) >= UStringTablePrivate::BATCHSIZE) {
        //lock.unlock();	// FIXME: okay to comment out?
        p->m_wait.notify_one();
    }
}

void UStringTable::flushRows()
{
    if (d->directWrite)
        return;

    int ret = 0;

    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_ustring select * from temp_rocpd_ustring", NULL, NULL, NULL);
    rpdLog("rocpd_ustring: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_ustring", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);

}

void UStringTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_slot->head() == m_slot->tail())
        return;

    //FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_slot->tail() + 1;
    int end = m_slot->tail() + BATCHSIZE;
    end = (end > m_slot->head()) ? m_slot->head() : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        // insert rocpd_string
        int index = 1;
        UStringTable::row &r = d->rows[i % m_slot->capacity()];
        //printf("%lld %s\n", r.string_id, r.string.c_str());
        sqlite3_bind_int64(d->stringInsert, index++, r.string_id + m_idOffset);
        sqlite3_bind_text(d->stringInsert, index++, r.string.c_str(), -1, SQLITE_STATIC);	// FIXME SQLITE_TRANSIENT?
        int ret = sqlite3_step(d->stringInsert);
        sqlite3_reset(d->stringInsert);
    }
    lock.lock();
    m_slot->tail() = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_end_time = clocktime_ns();
#if 0
    // FIXME
    if (done == false) {
        char buff[4096];
        std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_slot->head() - m_slot->tail());
        createOverheadRecord(cb_begin_time, cb_end_time, "UStringTable::writeRows", buff);
    }
#endif
}

}  // namespace rpdtracer
