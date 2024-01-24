/*********************************************************************************
* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
********************************************************************************/
#include "Table.h"

#include <thread>
#include <map>

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_MONITOR = "CREATE TEMPORARY TABLE \"temp_rocpd_monitor\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"deviceType\" varchar(16) NOT NULL, \"deviceId\" integer NOT NULL, \"monitorType\" varchar(16) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"value\" varchar(255) NOT NULL)";

class MonitorTablePrivate
{
public:
    MonitorTablePrivate(MonitorTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<MonitorTable::row, BUFFERSIZE> rows; // Circular buffer

    sqlite3_stmt *monitorInsert;

    class rowCompare
    {
    public:
        bool operator() (const MonitorTable::row& lhs, const MonitorTable::row& rhs) const
        {
            return lhs.deviceId < rhs.deviceId
                || lhs.monitorType < rhs.monitorType
                || lhs.deviceType < rhs.deviceType;
        }
    };

    std::map<MonitorTable::row, bool, rowCompare> values;
    void insertInternal(MonitorTable::row &row);

    MonitorTable *p;
};


MonitorTable::MonitorTable(const char *basefile)
: BufferedTable(basefile, MonitorTablePrivate::BUFFERSIZE, MonitorTablePrivate::BATCHSIZE)
, d(new MonitorTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_MONITOR, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) values (?,?,?,?,?,?)", -1, &d->monitorInsert, NULL);
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
    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) select deviceType, deviceId, monitorType, start, end, value from temp_rocpd_monitor", NULL, NULL, NULL);
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
        sqlite3_bind_text(d->monitorInsert, index++, r.value.c_str(), -1, SQLITE_STATIC);

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
