// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "BufferPool.h"

#include <thread>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::CopyApiTable;

namespace rpdtracer {

const char *SCHEMA_COPYAPI = "CREATE TEMPORARY TABLE \"temp_rocpd_copyapi\" (\"api_ptr_id\" integer NOT NULL PRIMARY KEY REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"stream\" varchar(18) NOT NULL, \"size\" integer NOT NULL, \"width\" integer NOT NULL, \"height\" integer NOT NULL, \"kind\" integer NOT NULL, \"dst\" varchar(18) NOT NULL, \"src\" varchar(18) NOT NULL, \"dstDevice\" integer NOT NULL, \"srcDevice\" integer NOT NULL, \"sync\" bool NOT NULL, \"pinned\" bool NOT NULL);";

class CopyApiTablePrivate
{
public:
    CopyApiTablePrivate(CopyApiTable *cls) : p(cls) {
        rows = p->m_slot->rows<CopyApiTable::row>();
    }
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    CopyApiTable::row *rows;

    sqlite3_stmt *apiInsert;
    bool directWrite;

    CopyApiTable *p;
};


CopyApiTable::CopyApiTable(const char *basefile, bool directWrite, BufferPool &pool)
: BufferedTable(basefile, pool.allocate<CopyApiTable::row>(CopyApiTablePrivate::BUFFERSIZE, "CopyApiTable"), CopyApiTablePrivate::BATCHSIZE)
, d(new CopyApiTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA_COPYAPI, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    }
}


CopyApiTable::~CopyApiTable()
{
    delete d;
}


void CopyApiTable::insert(const CopyApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_slot->head() - m_slot->tail() >= CopyApiTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }

    d->rows[(++m_slot->head()) % CopyApiTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_slot->head() - m_slot->tail()) >= CopyApiTablePrivate::BATCHSIZE) {
        lock.unlock();
        m_wait.notify_one();
    }
}


void CopyApiTable::flushRows()
{
    if (d->directWrite)
        return;

    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_copyapi select * from temp_rocpd_copyapi", NULL, NULL, NULL);
    rpdLog("rocpd_copyapi: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_copyapi", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void CopyApiTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_slot->head() == m_slot->tail())
        return;

    // FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_slot->tail() + 1;
    int end = m_slot->tail() + BATCHSIZE;
    end = (end > m_slot->head()) ? m_slot->head() : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        CopyApiTable::row &r = d->rows[i % m_slot->capacity()];

        sqlite3_bind_int64(d->apiInsert, index++, r.api_id + m_idOffset);
        sqlite3_bind_text(d->apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
        if (r.size > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.size);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.width > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.width);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.height > 0)
            sqlite3_bind_int(d->apiInsert, index++, r.height);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(d->apiInsert, index++, "", -1, SQLITE_STATIC);
        //sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int(d->apiInsert, index++, r.kind);
        sqlite3_bind_text(d->apiInsert, index++, r.dst.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(d->apiInsert, index++, r.src.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(d->apiInsert, index++, r.dstDevice);
        sqlite3_bind_int(d->apiInsert, index++, r.srcDevice);
        sqlite3_bind_int(d->apiInsert, index++, r.sync);
        sqlite3_bind_int(d->apiInsert, index++, r.pinned);
        int ret = sqlite3_step(d->apiInsert);
        sqlite3_reset(d->apiInsert);
    }
    lock.lock();
    m_slot->tail() = end;
    lock.unlock();

    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    // FIXME
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_slot->head() - m_slot->tail());
    createOverheadRecord(cb_begin_time, cb_end_time, "CopyApiTable::writeRows", buff);
}

}  // namespace rpdtracer
