#include "Table.h"
#include <thread>
#include <deque>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_API = "CREATE TEMPORARY TABLE \"temp_rocpd_api\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"pid\" integer NOT NULL, \"tid\" integer NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"apiName_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"args_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";


class ApiTablePrivate
{
public:
    ApiTablePrivate(ApiTable *cls) : p(cls) {}
    static const int BUFFERSIZE = 16384 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<ApiTable::row, BUFFERSIZE> rows; // Circular buffer
    int head;
    int tail;
    int count;

    std::map<sqlite3_int64, ApiTable::row> inFlight;
    std::map<std::pair<sqlite3_int64, sqlite3_int64>, std::deque<ApiTable::row>> roctxStacks;

    sqlite3_stmt *apiInsert;
    sqlite3_stmt *apiInsertNoId;

    void writeRows();

    void work();                // work thread
    std::thread *worker;
    bool workerRunning;
    bool done;

    ApiTable *p;
};


ApiTable::ApiTable(const char *basefile)
: Table(basefile)
, d(new ApiTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_API, NULL, NULL, NULL);

    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api(pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?)", -1, &d->apiInsertNoId, NULL);

    d->head = 0;
    d->tail = 0;

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->worker = new std::thread(&ApiTablePrivate::work, d);
}

void ApiTable::insert(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
	if (row.phase == 0) {
       d->inFlight.insert({row.api_id, row});
       return;
    }

    if (d->head - d->tail >= ApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        printf("Trouble\n");
        // FIXME: overhead record here
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    if (row.phase == 1) {
        auto it = d->inFlight.find(row.api_id);
        if (it != d->inFlight.end()) {
            ApiTable::row &r = it->second;
            r.end = row.end;
            d->rows[(++d->head) % ApiTablePrivate::BUFFERSIZE] = r;
            d->inFlight.erase(it);
        }
    }

    if (d->workerRunning == false && (d->head - d->tail) >= ApiTablePrivate::BATCHSIZE)
        m_wait.notify_one();
}

static sqlite3_int64 roctx_id_hack = sqlite3_int64(1) << 31;

void ApiTable::insertRoctx(ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (d->head - d->tail >= ApiTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }
    row.api_id = ++roctx_id_hack;
    d->rows[(++d->head) % ApiTablePrivate::BUFFERSIZE] = row;

    if (d->workerRunning == false && (d->head - d->tail) >= ApiTablePrivate::BATCHSIZE)
        m_wait.notify_one();
}

void ApiTable::pushRoctx(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto key = std::pair<sqlite3_int64, sqlite3_int64>(row.pid, row.tid);
    auto &stack = d->roctxStacks[key];
    stack.push_front(row);
}

void ApiTable::popRoctx(const ApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (d->head - d->tail >= ApiTablePrivate::BUFFERSIZE) {
        m_wait.notify_one();
        m_wait.wait(lock);
    }
    auto key = std::pair<sqlite3_int64, sqlite3_int64>(row.pid, row.tid);
    auto &stack = d->roctxStacks[key];
    if (stack.empty() == false) {
        ApiTable::row &r = stack.front();
        r.end = row.end;
        r.api_id = ++roctx_id_hack;
        d->rows[(++d->head) % ApiTablePrivate::BUFFERSIZE] = r;
        stack.pop_front();
    }

    if (d->workerRunning == false && (d->head - d->tail) >= ApiTablePrivate::BATCHSIZE)
        m_wait.notify_one();
}

void ApiTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void ApiTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_api select * from temp_rocpd_api", NULL, NULL, NULL);
    printf("rocpd_api: %d\n", ret);
}


void ApiTablePrivate::writeRows()
{
    int i = 1;

    std::unique_lock<std::mutex> lock(p->m_mutex);

    if (head == tail)
        return;

    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    int start = tail + 1;
    int end = tail + BATCHSIZE;
    end = (end > head) ? head : end;
    lock.unlock();

    for (i = start; i <= end; ++i) {
        // insert rocpd_api
        int index = 1;
        ApiTable::row &r = rows[i % BUFFERSIZE];
        sqlite3_bind_int64(apiInsert, index++, r.api_id + p->m_idOffset);
        sqlite3_bind_int(apiInsert, index++, r.pid);
        sqlite3_bind_int(apiInsert, index++, r.tid);
        sqlite3_bind_int64(apiInsert, index++, r.start);
        sqlite3_bind_int64(apiInsert, index++, r.end);
        sqlite3_bind_int64(apiInsert, index++, r.apiName_id + p->m_idOffset);
        sqlite3_bind_int64(apiInsert, index++, r.args_id + p->m_idOffset);
        int ret = sqlite3_step(apiInsert);
        sqlite3_reset(apiInsert);
    }
    lock.lock();
    tail = end;
    lock.unlock();

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME: write the overhead record
}


void ApiTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= ApiTablePrivate::BATCHSIZE) {
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


