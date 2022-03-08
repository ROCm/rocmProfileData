#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_KERNELAPI = "CREATE TEMPORARY TABLE \"temp_rocpd_kernelapi\" (\"api_ptr_id\" integer NOT NULL PRIMARY KEY, \"stream\" varchar(18) NOT NULL, \"gridX\" integer NOT NULL, \"gridY\" integer NOT NULL, \"gridz\" integer NOT NULL, \"workgroupX\" integer NOT NULL, \"workgroupY\" integer NOT NULL, \"workgroupZ\" integer NOT NULL, \"groupSegmentSize\" integer NOT NULL, \"privateSegmentSize\" integer NOT NULL, \"kernelArgAddress\" varchar(18) NOT NULL, \"aquireFence\" varchar(8) NOT NULL, \"releaseFence\" varchar(8) NOT NULL, \"codeObject_id\" integer, \"kernelName_id\" integer NOT NULL)";

class KernelApiTablePrivate
{
public:
    KernelApiTablePrivate(KernelApiTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 16384;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<KernelApiTable::row, BUFFERSIZE> rows; // Circular buffer
    int head;
    int tail;
    int count;

    sqlite3_stmt *apiInsert;

    void writeRows();

    void work();		// work thread
    std::thread *worker;
    bool workerRunning;
    bool done;

    KernelApiTable *p;
};


KernelApiTable::KernelApiTable(const char *basefile)
: Table(basefile)
, d(new KernelApiTablePrivate(this))
{
    int ret;
    // set up tmp table
    ret = sqlite3_exec(m_connection, SCHEMA_KERNELAPI, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, kernelArgAddress, aquireFence, releaseFence, codeObject_id, kernelName_id) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", -1, &d->apiInsert, NULL);
    
    d->head = 0;    // last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->worker = new std::thread(&KernelApiTablePrivate::work, d);
}

void KernelApiTable::insert(const KernelApiTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (d->head - d->tail >= KernelApiTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++d->head) % KernelApiTablePrivate::BUFFERSIZE] = row;

    if (d->workerRunning == false && (d->head - d->tail) >= KernelApiTablePrivate::BATCHSIZE) {
        m_wait.notify_one();
    }
}

void KernelApiTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void KernelApiTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_kernelapi select * from temp_rocpd_kernelapi", NULL, NULL, NULL);
    printf("rocpd_kernelapi: %d\n", ret);
}


void KernelApiTablePrivate::writeRows()
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
        KernelApiTable::row &r = rows[(tail + i) % BUFFERSIZE];
        sqlite3_bind_int64(apiInsert, index++, r.api_id + p->m_idOffset);
        sqlite3_bind_text(apiInsert, index++, r.stream.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(apiInsert, index++, r.gridX);
        sqlite3_bind_int(apiInsert, index++, r.gridY);
        sqlite3_bind_int(apiInsert, index++, r.gridZ);
        sqlite3_bind_int(apiInsert, index++, r.workgroupX);
        sqlite3_bind_int(apiInsert, index++, r.workgroupY);
        sqlite3_bind_int(apiInsert, index++, r.workgroupZ);
        sqlite3_bind_int(apiInsert, index++, r.groupSegmentSize);
        sqlite3_bind_int(apiInsert, index++, r.privateSegmentSize);
        sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_text(apiInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int64(apiInsert, index++, r.kernelName_id + p->m_idOffset);
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


void KernelApiTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= KernelApiTablePrivate::BATCHSIZE) {
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
