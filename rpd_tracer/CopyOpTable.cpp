#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_COPYOP = "CREATE TEMPORARY TABLE \"temp_rocpd_copyop\" (\"op_ptr_id\" integer NOT NULL PRIMARY KEY, \"size\" integer NOT NULL, \"src\" integer NOT NULL, \"dst\" integer NOT NULL, \"sync\" bool NOT NULL, \"pinned\" bool NOT NULL);";

class CopyOpTablePrivate
{
public:
    CopyOpTablePrivate(CopyOpTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 16384;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<CopyOpTable::row, BUFFERSIZE> rows; // Circular buffer
    int head;
    int tail;
    int count;

    sqlite3_stmt *opInsert;

    void writeRows();

    void work();		// work thread
    std::thread *worker;
    bool workerRunning;
    bool done;

    CopyOpTable *p;
};


CopyOpTable::CopyOpTable(const char *basefile)
: Table(basefile)
, d(new CopyOpTablePrivate(this))
{
    int ret;
    // set up tmp table
    ret = sqlite3_exec(m_connection, SCHEMA_COPYOP, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_copyop(op_ptr_id, size, src, dst, sync, pinned), values (?,?,?,?,?,?)", -1, &d->opInsert, NULL);
    
    d->head = 0;    // last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->worker = new std::thread(&CopyOpTablePrivate::work, d);
}

void CopyOpTable::insert(const CopyOpTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (d->head - d->tail >= CopyOpTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++d->head) % CopyOpTablePrivate::BUFFERSIZE] = row;

    if (d->workerRunning == false && (d->head - d->tail) >= CopyOpTablePrivate::BATCHSIZE) {
        m_wait.notify_one();
    }
}

void CopyOpTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void CopyOpTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_copyop select * from temp_rocpd_copyop", NULL, NULL, NULL);
    printf("rocpd_copyop: %d\n", ret);
}


void CopyOpTablePrivate::writeRows()
{
    int i = 1;

    std::unique_lock<std::mutex> guard(p->m_mutex);

    if (head == tail)
        return;

    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    while (i < BATCHSIZE && (head > tail + i)) {	// FIXME: refactor like ApiTable?
        // insert rocpd_op
        int index = 1;
        CopyOpTable::row &r = rows[(tail + i) % BUFFERSIZE];

        sqlite3_bind_int64(opInsert, index++, r.op_id);
        sqlite3_bind_int64(opInsert, index++, r.size);
        //sqlite3_bind_int(opInsert, index++, r.width);
        //sqlite3_bind_int(opInsert, index++, r.height);
        //sqlite3_bind_text(opInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_text(opInsert, index++, r.src.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(opInsert, index++, r.dst.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(opInsert, index++, r.sync);
        sqlite3_bind_int(opInsert, index++, r.pinned);
        int ret = sqlite3_step(opInsert);
        sqlite3_reset(opInsert);
        ++i;
    }
    tail = tail + i;

    guard.unlock();

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
}


void CopyOpTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= CopyOpTablePrivate::BATCHSIZE) {
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
