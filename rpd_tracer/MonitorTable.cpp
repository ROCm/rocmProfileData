// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"

#include <thread>
#include <map>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::MonitorTable;

namespace rpdtracer {

const char *SCHEMA_MONITOR = "CREATE TEMPORARY TABLE \"temp_rocpd_monitor\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"deviceType\" varchar(16) NOT NULL, \"deviceId\" integer NOT NULL, \"monitorType\" varchar(16) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"value\" integer NOT NULL)";

class MonitorTablePrivate
{
public:
    MonitorTablePrivate(MonitorTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<MonitorTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *monitorInsert;
    bool directWrite;

    class rowCompare
    {
    public:
        bool operator() (const MonitorTable::row& lhs, const MonitorTable::row& rhs) const
        {
            if (lhs.deviceId != rhs.deviceId) return lhs.deviceId < rhs.deviceId;
            if (lhs.monitorType != rhs.monitorType) return lhs.monitorType < rhs.monitorType;
            return lhs.deviceType < rhs.deviceType;
        }
    };

    std::map<MonitorTable::row, bool, rowCompare> values;
    void insertInternal(MonitorTable::row &row);

    MonitorTable *p;
};


MonitorTable::MonitorTable(const char *basefile, bool directWrite)
: BufferedTable(basefile, MonitorTablePrivate::BUFFERSIZE, MonitorTablePrivate::BATCHSIZE)
, d(new MonitorTablePrivate(this))
{
    int ret;
    d->directWrite = directWrite;

    if (!directWrite) {
        ret = sqlite3_exec(m_connection, SCHEMA_MONITOR, NULL, NULL, NULL);
        ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) values (?,?,?,?,?,?)", -1, &d->monitorInsert, NULL);
    } else {
        ret = sqlite3_prepare_v2(m_connection, "insert into rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) values (?,?,?,?,?,?)", -1, &d->monitorInsert, NULL);
    }
}


MonitorTable::~MonitorTable()
{
    delete d;
}


void MonitorTable::insert(const MonitorTable::row &row)
{
    auto it = d->values.find(row);
    if (it == d->values.end()) {
        d->values.insert(std::pair<MonitorTable::row, bool>(row, true));
        it = d->values.find(row);
    }
    MonitorTable::row &old = const_cast<MonitorTable::row&>((*it).first);  // Oh yes

#if 0
    static sqlite3_int64 prev = clocktime_ns();
    fprintf(stderr, "      %lld   delta %lld\n", row.start, row.start - prev);
    prev = row.start;
#endif

    if (old.value != row.value) {  // value changed, actually insert a row
        old.end = row.start;
        //fprintf(stderr, "+++++ %lld\n", row.start);
        d->insertInternal(old);
        old = row;
    }
}

void MonitorTablePrivate::insertInternal(MonitorTable::row &row)
{
    std::unique_lock<std::mutex> lock(p->m_mutex);
    if (p->m_head - p->m_tail >= MonitorTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        const timestamp_t start = clocktime_ns();
        p->m_wait.notify_one();  // make sure working is running
        p->m_wait.wait(lock);

        const timestamp_t end = clocktime_ns();
        lock.unlock();
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::MonitorTable::insert");
        lock.lock();
    }

    rows[(++(p->m_head)) % MonitorTablePrivate::BUFFERSIZE] = row;

    if (p->workerRunning() == false && (p->m_head - p->m_tail) >= MonitorTablePrivate::BATCHSIZE) {
        lock.unlock();
        p->m_wait.notify_one();
    }
}

void MonitorTable::endCurrentRuns(sqlite3_int64 endTimestamp)
{
    for (auto it = d->values.begin(); it != d->values.end(); ++it) {
        MonitorTable::row &old = const_cast<MonitorTable::row&>((*it).first);  // Oh yes
        old.end = endTimestamp;
        //fprintf(stderr, "final %lld\n", old.end);
        d->insertInternal(old);
    }
    d->values.clear();
}


void MonitorTable::flushRows()
{
    if (d->directWrite)
        return;

    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) select deviceType, deviceId, monitorType, start, end, value from temp_rocpd_monitor", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_monitor: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_monitor", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void MonitorTable::writeRows()
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

    sqlite3_exec(m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        int index = 1;
        MonitorTable::row &r = d->rows[i % BUFFERSIZE];
        sqlite3_bind_text(d->monitorInsert, index++, r.deviceType.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(d->monitorInsert, index++, r.deviceId);
        sqlite3_bind_text(d->monitorInsert, index++, r.monitorType.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(d->monitorInsert, index++, r.start);
        sqlite3_bind_int64(d->monitorInsert, index++, r.end);
        sqlite3_bind_int64(d->monitorInsert, index++, r.value);

        int ret = sqlite3_step(d->monitorInsert);
        sqlite3_reset(d->monitorInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "MonitorTable::writeRows", buff);
}

}  // namespace rpdtracer
