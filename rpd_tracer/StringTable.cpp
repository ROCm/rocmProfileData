/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Table.h"

#include <thread>
#include <unordered_map>

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_STRING = "CREATE TEMPORARY TABLE \"temp_rocpd_string\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"string\" varchar(4096) NOT NULL)";


class StringTablePrivate
{
public:
    StringTablePrivate(StringTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 8;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<StringTable::row, BUFFERSIZE> rows; // Circular buffer
    //std::map<std::string,sqlite3_int64> cache;     // Cache for string lookups
    std::unordered_map<std::string,sqlite3_int64> cache;     // Cache for string lookups
    int head;
    int tail;
    int count;

    sqlite3_stmt *stringInsert;

    void insert(StringTable::row&);
    void writeRows();

    void work();		// work thread
    std::thread *worker;
    bool done;
    bool workerRunning;
    std::mutex cacheMutex;

    StringTable *p;
};


StringTable::StringTable(const char *basefile)
: Table(basefile)
, d(new StringTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_STRING, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_string(id, string) values (?,?)", -1, &d->stringInsert, NULL);
    
    d->head = 0;	// last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->worker = new std::thread(&StringTablePrivate::work, d);
    d->cache.reserve(64 * 1024);  // Avoid/delay rehashing for typical runs

    StringTable::getOrCreate("");    // empty string is id=1
}

sqlite3_int64 StringTable::getOrCreate(const std::string &key)
{
    std::lock_guard<std::mutex> guard(d->cacheMutex);
    auto it = d->cache.find(key);
    if (it == d->cache.end()) {
        // new string, create a row
        StringTable::row row;
        row.string_id = 0;
        row.string = key;
        d->insert(row);		// string_id gets updated with id
        // update cache
        d->cache.insert({row.string, row.string_id});
        return row.string_id;
    }
    return it->second;
}

void StringTablePrivate::insert(StringTable::row &row)
{
    std::unique_lock<std::mutex> lock(p->m_mutex);
    if (head - tail >= StringTablePrivate::BUFFERSIZE) {
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
        createOverheadRecord(start, end, "BLOCKING", "rpd_tracer::StringTable::insert");
        lock.lock();
    }

    row.string_id = ++head;
    rows[head % StringTablePrivate::BUFFERSIZE] = row;

    if (workerRunning == false && (head - tail) >= StringTablePrivate::BATCHSIZE) {
        lock.unlock();
        p->m_wait.notify_one();
    }
}

void StringTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void StringTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;

    flush();
    int ret = 0;
    // FIXME: to empty string or not to empty string?  multi-session issue?
    //ret = sqlite3_exec(m_connection, "insert into rocpd_string select * from temp_rocpd_string where id != 1", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_string select * from temp_rocpd_string", NULL, NULL, NULL);
    printf("rocpd_string: %d\n", ret);
}


void StringTablePrivate::writeRows()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    if (head == tail)
        return;

    //const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_begin_time = clocktime_ns();

    int start = tail + 1;
    int end = tail + BATCHSIZE;
    end = (end > head) ? head : end;
    lock.unlock();

    sqlite3_exec(p->m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    for (int i = start; i <= end; ++i) {
        // insert rocpd_string
        int index = 1;
        StringTable::row &r = rows[i % BUFFERSIZE];
        //printf("%lld %s\n", r.string_id, r.string.c_str());
        sqlite3_bind_int64(stringInsert, index++, r.string_id + p->m_idOffset);
        sqlite3_bind_text(stringInsert, index++, r.string.c_str(), -1, SQLITE_STATIC);	// FIXME SQLITE_TRANSIENT?
        int ret = sqlite3_step(stringInsert);
        sqlite3_reset(stringInsert);
    }
    lock.lock();
    tail = end;
    lock.unlock();

    //const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    //const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    //FIXME
    const timestamp_t cb_end_time = clocktime_ns();
    if (done == false) {
        char buff[4096];
        std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, head - tail);
        createOverheadRecord(cb_begin_time, cb_end_time, "StringTable::writeRows", buff);
    }
}


void StringTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= StringTablePrivate::BATCHSIZE) {
            lock.unlock();
            writeRows();
            p->m_wait.notify_all();
            lock.lock();
        }
        workerRunning = false;
        if (done == false)
            p->m_wait.wait(lock);
        workerRunning = true;
    }
}
