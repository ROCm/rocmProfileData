/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


const char *SCHEMA_OP = "CREATE TEMPORARY TABLE \"temp_rocpd_op\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"gpuId\" integer NOT NULL, \"queueId\" integer NOT NULL, \"sequenceId\" integer NOT NULL, \"completionSignal\" varchar(18) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"description_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"opType_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";

const char *SCHEMA_API_OPS = "CREATE TEMPORARY TABLE \"temp_rocpd_api_ops\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"api_id\" integer NOT NULL REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"op_id\" integer NOT NULL REFERENCES \"rocpd_op\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";


class OpTablePrivate
{
public:
    OpTablePrivate(OpTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 16384;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<OpTable::row, BUFFERSIZE> rows; // Circular buffer
    std::map<sqlite3_int64, sqlite3_int64> descriptions;
    std::mutex m_descriptionLock;
    int head;
    int tail;
    int count;

    std::map<sqlite3_int64, KernelOpTable::row> kernelRows;
    std::map<sqlite3_int64, CopyOpTable::row> copyRows;

    sqlite3_stmt *opInsert;
    sqlite3_stmt *apiOpInsert;

    void writeRows();

    void work();		// work thread
    std::thread *worker;
    bool workerRunning;
    bool done;

    OpTable *p;
    KernelOpTable *kp;
    CopyOpTable *cp;
};


OpTable::OpTable(const char *basefile, KernelOpTable *kernelTable, CopyOpTable *copyTable)
: Table(basefile)
, d(new OpTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_OP, NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, SCHEMA_API_OPS, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_op(id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?,?)", -1, &d->opInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api_ops(api_id, op_id) values (?,?)", -1, &d->apiOpInsert, NULL);
    
    d->head = 0;    // last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
    d->done = false;
    d->workerRunning = true;

    d->kp = kernelTable;
    d->cp = copyTable;

    d->worker = new std::thread(&OpTablePrivate::work, d);
}

void OpTable::insert(const OpTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (d->head - d->tail >= OpTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++d->head) % OpTablePrivate::BUFFERSIZE] = row;

    if (d->workerRunning == false && (d->head - d->tail) >= OpTablePrivate::BATCHSIZE) {
        m_wait.notify_one();
    }
}

void OpTable::associateDescription(const sqlite3_int64 &api_id, const sqlite3_int64 &string_id)
{
    // FIXME: remove this, kernelops has kernel description
    // FIXME: double buffer this?
    // writeRows uses this structure heavily, intermittently.  May matter
    std::lock_guard<std::mutex> guard(d->m_descriptionLock);
    d->descriptions[api_id] = string_id;
}

void OpTable::associateKernelOp(const sqlite3_int64 &api_id, KernelOpTable::row krow)
{
    // FIXME: double buffer this?
    // writeRows uses this structure heavily, intermittently.  May matter
    d->kernelRows.emplace(std::make_pair(api_id, krow));
}

void OpTable::associateCopyOp(const sqlite3_int64 &api_id, CopyOpTable::row crow)
{
    // FIXME: double buffer this?
    // writeRows uses this structure heavily, intermittently.  May matter
    d->copyRows.emplace(std::make_pair(api_id, crow));
}

void OpTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void OpTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    delete d->worker;
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_op select * from temp_rocpd_op", NULL, NULL, NULL);
    printf("rocpd_op: %d\n", ret);
    ret = sqlite3_exec(m_connection, "insert into rocpd_api_ops (api_id, op_id) select api_id, op_id from temp_rocpd_api_ops", NULL, NULL, NULL);
    printf("rocpd_api_ops: %d\n", ret);
}


void OpTablePrivate::writeRows()
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
        // insert rocpd_op
        int index = 1;
        OpTable::row &r = rows[i % BUFFERSIZE];
        sqlite3_int64 primaryKey = i + p->m_idOffset;

        // check for description override
#if 0
        {
            std::lock_guard<std::mutex> guard(m_descriptionLock);
            auto it = descriptions.find(r.api_id);
            if (it != descriptions.end()) {
                r.description_id = it->second;
                descriptions.erase(it);
            }
        }
#endif
        // Check for kernelop
        {
            std::lock_guard<std::mutex> guard(m_descriptionLock);
            auto it = kernelRows.find(r.api_id);
            if (it != kernelRows.end()) {
                KernelOpTable::row &krow = it->second;
                krow.op_id = primaryKey;
                kp->insert(krow);
                //printf("krow for %lld\n", primaryKey);
                kernelRows.erase(it);
                // Copy kernel name into description of base op (for historical reasons)
                r.description_id = krow.kernelName_id;
            }
        }

        sqlite3_bind_int64(opInsert, index++, primaryKey);
        sqlite3_bind_int(opInsert, index++, r.gpuId);
        sqlite3_bind_int(opInsert, index++, r.queueId);
        sqlite3_bind_int(opInsert, index++, r.sequenceId);
        sqlite3_bind_text(opInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int64(opInsert, index++, r.start);
        sqlite3_bind_int64(opInsert, index++, r.end);
        sqlite3_bind_int64(opInsert, index++, r.description_id + p->m_idOffset);
        sqlite3_bind_int64(opInsert, index++, r.opType_id + p->m_idOffset);
        int ret = sqlite3_step(opInsert);
        sqlite3_reset(opInsert);

        // Insert rocpd_api_ops
        //sqlite_int64 rowId = sqlite3_last_insert_rowid(p->m_connection);
        index = 1;
        sqlite3_bind_int64(apiOpInsert, index++, sqlite3_int64(r.api_id) + p->m_idOffset);
        sqlite3_bind_int64(apiOpInsert, index++, sqlite3_int64(i) + p->m_idOffset);
        ret = sqlite3_step(apiOpInsert);
        sqlite3_reset(apiOpInsert);
    }
    lock.lock();
    tail = end;
    lock.unlock();

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    // FIXME: write the overhead record
}


void OpTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((head - tail) >= OpTablePrivate::BATCHSIZE) {
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
