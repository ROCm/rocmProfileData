// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"

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

class KernelApiTablePrivate
{
public:
    KernelApiTablePrivate(KernelApiTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<KernelApiTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *apiInsert;
    bool directWrite;

    KernelApiTable *p;
};


KernelApiTable::KernelApiTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, KernelApiTablePrivate::BUFFERSIZE, KernelApiTablePrivate::BATCHSIZE)
, d(new KernelApiTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA_KERNELAPI, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, kernelName_id) values (?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, kernelName_id) values (?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    }
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
    if (d->directWrite)
        return;

    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_kernelapi select * from temp_rocpd_kernelapi", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_kernelapi: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_kernelapi", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void KernelApiTable::writeRows()
{
    std::unique_lock<std::mutex> wlock(m_writeMutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_head == m_tail)
        return;

    //const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = m_tail + 1;
    int end = m_tail + BATCHSIZE;
    end = (end > m_head) ? m_head : end;
    lock.unlock();

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        KernelApiTable::row &r = d->rows[i % BUFFERSIZE];
        sqlite3_bind_int64(d->apiInsert, index++, r.api_id + m_idOffset);
        sqlite3_bind_text(d->apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(d->apiInsert, index++, r.gridX);
        sqlite3_bind_int(d->apiInsert, index++, r.gridY);
        sqlite3_bind_int(d->apiInsert, index++, r.gridZ);
        sqlite3_bind_int(d->apiInsert, index++, r.workgroupX);
        sqlite3_bind_int(d->apiInsert, index++, r.workgroupY);
        sqlite3_bind_int(d->apiInsert, index++, r.workgroupZ);
        sqlite3_bind_int(d->apiInsert, index++, r.groupSegmentSize);
        sqlite3_bind_int(d->apiInsert, index++, r.privateSegmentSize);
        sqlite3_bind_int64(d->apiInsert, index++, r.kernelName_id + m_idOffset);
        int ret = sqlite3_step(d->apiInsert);
        sqlite3_reset(d->apiInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME
    const timestamp_t cb_end_time = clocktime_ns() + 1;
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "KernelApiTable::writeRows", buff);
}

}  // namespace rpdtracer
