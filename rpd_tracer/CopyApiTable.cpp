#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_COPYAPI = "CREATE TEMPORARY TABLE \"temp_rocpd_copyapi\" (\"api_ptr_id\" integer NOT NULL PRIMARY KEY REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"stream\" varchar(18) NOT NULL, \"size\" integer NOT NULL, \"width\" integer NOT NULL, \"height\" integer NOT NULL, \"kind\" integer NOT NULL, \"dst\" varchar(18) NOT NULL, \"src\" varchar(18) NOT NULL, \"dstDevice\" integer NOT NULL, \"srcDevice\" integer NOT NULL, \"sync\" bool NOT NULL, \"pinned\" bool NOT NULL);";

class CopyApiTablePrivate
{
public:
    CopyApiTablePrivate(CopyApiTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 16384;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<CopyApiTable::row, BUFFERSIZE> rows; // Circular buffer
    int head;
    int tail;
    int count;

    sqlite3_stmt *apiInsert;

    void writeRows();

    void work();		// work thread
    std::thread *worker;
    bool workerRunning;
    bool done;

    CopyApiTable *p;
};


CopyApiTable::CopyApiTable(const char *basefile)
: Table(basefile)
, d(new CopyApiTablePrivate(this))
{
    int ret;
    // set up tmp table
    ret = sqlite3_exec(m_connection, SCHEMA_COPYAPI, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    
    d->head = 0;    // last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->worker = new std::thread(&CopyApiTablePrivate::work, d);
}

void CopyApiTable::insert(const CopyApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (d->head - d->tail >= CopyApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++d->head) % CopyApiTablePrivate::BUFFERSIZE] = row;

    if (d->workerRunning == false && (d->head - d->tail) >= CopyApiTablePrivate::BATCHSIZE) {
        m_wait.notify_one();
    }
}

void CopyApiTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void CopyApiTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_copyapi select * from temp_rocpd_copyapi", NULL, NULL, NULL);
    printf("rocpd_copyapi: %d\n", ret);
}


void CopyApiTablePrivate::writeRows()
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
        CopyApiTable::row &r = rows[(tail + i) % BUFFERSIZE];

        sqlite3_bind_int64(apiInsert, index++, r.api_id + p->m_idOffset);
        sqlite3_bind_text(apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
        if (r.size > 0)
            sqlite3_bind_int(apiInsert, index++, r.size);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.width > 0)
            sqlite3_bind_int(apiInsert, index++, r.width);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        if (r.height > 0)
            sqlite3_bind_int(apiInsert, index++, r.height);
        else
            //sqlite3_bind_null(apiInsert, index++);
            sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        //sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int(apiInsert, index++, r.kind);
        sqlite3_bind_text(apiInsert, index++, r.dst.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(apiInsert, index++, r.src.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(apiInsert, index++, r.dstDevice);
        sqlite3_bind_int(apiInsert, index++, r.srcDevice);
        sqlite3_bind_int(apiInsert, index++, r.sync);
        sqlite3_bind_int(apiInsert, index++, r.pinned);
        int ret = sqlite3_step(apiInsert);
        sqlite3_reset(apiInsert);
        ++i;
    }
    tail = tail + i;

    guard.unlock();

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
}


void CopyApiTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= CopyApiTablePrivate::BATCHSIZE) {
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
