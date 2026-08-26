// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "NetWriterBackend.h"

#include <thread>
#include <map>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"

using rpdtracer::MonitorTable;

namespace rpdtracer {

const char *SCHEMA_MONITOR = "CREATE TEMPORARY TABLE \"temp_rocpd_monitor\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"deviceType\" varchar(16) NOT NULL, \"deviceId\" integer NOT NULL, \"monitorType\" varchar(16) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"value\" integer NOT NULL)";


class MonitorTableWriterBackend : public WriterBackend
{
public:
    MonitorTableWriterBackend(const char *basefile, bool directWrite)
    : m_directWrite(directWrite)
    {
        rpdSqliteOpen(basefile, &m_conn);
        sqlite3_busy_handler(m_conn, &rpdtracer::sqlite_busy_handler, NULL);
        sqlite3_exec(m_conn, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
        sqlite3_exec(m_conn, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

        if (!directWrite) {
            sqlite3_exec(m_conn, SCHEMA_MONITOR, NULL, NULL, NULL);
            sqlite3_prepare_v2(m_conn, "insert into temp_rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) values (?,?,?,?,?,?)", -1, &m_monitorInsert, NULL);
        } else {
            sqlite3_prepare_v2(m_conn, "insert into rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) values (?,?,?,?,?,?)", -1, &m_monitorInsert, NULL);
        }
    }

    ~MonitorTableWriterBackend() {
        sqlite3_finalize(m_monitorInsert);
        sqlite3_close(m_conn);
    }

    void setIdOffset(sqlite3_int64 offset) override { m_idOffset = offset; }

    void writeBatch(void *rowData, int start, int end, int capacity) override {
        auto *rows = static_cast<MonitorTable::row*>(rowData);

        sqlite3_exec(m_conn, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

        for (int i = start; i <= end; ++i) {
            int index = 1;
            MonitorTable::row &r = rows[i % capacity];
            sqlite3_bind_text(m_monitorInsert, index++, r.deviceType.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(m_monitorInsert, index++, r.deviceId);
            sqlite3_bind_text(m_monitorInsert, index++, r.monitorType.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(m_monitorInsert, index++, r.start);
            sqlite3_bind_int64(m_monitorInsert, index++, r.end);
            sqlite3_bind_int64(m_monitorInsert, index++, r.value);

            sqlite3_step(m_monitorInsert);
            sqlite3_reset(m_monitorInsert);
        }

        sqlite3_exec(m_conn, "END TRANSACTION", NULL, NULL, NULL);
    }

    void flush() override {
        if (m_directWrite)
            return;
        int ret = 0;
        ret = sqlite3_exec(m_conn, "begin transaction", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "insert into rocpd_monitor(deviceType, deviceId, monitorType, start, end, value) select deviceType, deviceId, monitorType, start, end, value from temp_rocpd_monitor", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "delete from temp_rocpd_monitor", NULL, NULL, NULL);
        ret = sqlite3_exec(m_conn, "commit", NULL, NULL, NULL);
    }

private:
    sqlite3 *m_conn;
    sqlite3_stmt *m_monitorInsert;
    sqlite3_int64 m_idOffset{0};
    bool m_directWrite;
};

WriterBackend* MonitorTable::createWriterBackend(const char *basefile, bool directWrite)
{
    return new MonitorTableWriterBackend(basefile, directWrite);
}

static void serializeMonitorTableRow(const void *row, ByteBuffer &buf) {
    static_cast<const MonitorTable::row*>(row)->serialize(buf);
}

WriterBackend* MonitorTable::createNetWriterBackend(const char *host, int port, bool directWrite)
{
    return new NetWriterBackend("MonitorTable", host, port, directWrite,
        sizeof(MonitorTable::row), serializeMonitorTableRow);
}


class MonitorTablePrivate
{
public:
    MonitorTablePrivate(MonitorTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<MonitorTable::row, BUFFERSIZE> rows; // Circular buffer

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
: BufferedTable(basefile, MonitorTablePrivate::BUFFERSIZE, MonitorTablePrivate::BATCHSIZE,
    isRemoteNode() ? createNetWriterBackend(getLogaggHost(), getLogaggPort(), directWrite)
                   : createWriterBackend(basefile, directWrite))
, d(new MonitorTablePrivate(this))
{
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
    rpdLog("      %lld   delta %lld\n", row.start, row.start - prev);
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
    m_writerBackend->flush();
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

    m_writerBackend->writeBatch(d->rows.data(), start, end, BUFFERSIZE);

    lock.lock();
    m_tail = end;
    lock.unlock();

    const timestamp_t cb_end_time = clocktime_ns();
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "MonitorTable::writeRows", buff);
}


void MonitorTable::row::serialize(ByteBuffer &buf) const {
    buf.writeString(deviceType);
    buf.writeString(monitorType);
    buf.writeInt64(deviceId);
    buf.writeInt64(start);
    buf.writeInt64(end);
    buf.writeInt64(value);
}

void MonitorTable::row::deserialize(ByteBuffer &buf) {
    deviceType = buf.readString();
    monitorType = buf.readString();
    deviceId = buf.readInt64();
    start = buf.readInt64();
    end = buf.readInt64();
    value = buf.readInt64();
}

}  // namespace rpdtracer
