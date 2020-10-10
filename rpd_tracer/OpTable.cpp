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
    int head;
    int tail;
    int count;

    sqlite3_stmt *opInsert;
    sqlite3_stmt *apiOpInsert;

    void writeRows();

    void work();		// work thread
    std::thread *worker;

    OpTable *p;
};


OpTable::OpTable(const char *basefile)
: Table(basefile)
, d(new OpTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_OP, NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, SCHEMA_API_OPS, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_op(gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)", -1, &d->opInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api_ops(api_id, op_id) values (?,?)", -1, &d->apiOpInsert, NULL);
    
    d->head = 0;	// last produced by insert()
    d->tail = 0;    // last consumed by 

    d->worker = NULL;
}

void OpTable::insert(const OpTable::row &row)
{
    if (d->head - d->tail >= OpTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        d->writeRows();
    }
    d->rows[++d->head] = row;
    if ((d->head - d->tail) >= OpTablePrivate::BATCHSIZE) {
        d->writeRows();
    }
}

void OpTable::flush()
{
    while (d->head > d->tail)
        d->writeRows();
}

void OpTable::finalize()
{
    flush();
    int ret = 0;
    ret = sqlite3_exec(m_connection, "insert into rocpd_op select * from temp_rocpd_op", NULL, NULL, NULL);
    printf("rocpd_op: %d\n", ret);
    ret = sqlite3_exec(m_connection, "insert into rocpd_api_ops select * from temp_rocpd_api_ops", NULL, NULL, NULL);
    printf("rocpd_api_ops: %d\n", ret);
}


void OpTablePrivate::writeRows()
{
    int i = 0;

    if (head == tail)
        return;

    const timestamp_t cb_begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "BEGIN DEFERRED TRANSACTION", NULL, NULL, NULL);

    while (i < BATCHSIZE && (head > tail + i)) {
        // insert rocpd_op
        int index = 1;
        OpTable::row &r = rows[tail + i];
        sqlite3_bind_int(opInsert, index++, r.gpuId);
        sqlite3_bind_int(opInsert, index++, r.queueId);
        sqlite3_bind_int(opInsert, index++, r.sequenceId);
        sqlite3_bind_text(opInsert, index++, "", -1, SQLITE_STATIC);
        sqlite3_bind_int64(opInsert, index++, r.start);
        sqlite3_bind_int64(opInsert, index++, r.end);
        sqlite3_bind_int64(opInsert, index++, r.description_id);
        sqlite3_bind_int64(opInsert, index++, r.opType_id);
        int ret = sqlite3_step(opInsert);
        sqlite3_reset(opInsert);

        // Insert rocpd_api_ops
        sqlite_int64 rowId = sqlite3_last_insert_rowid(p->m_connection);
        index = 1;
        sqlite3_bind_int64(apiOpInsert, index++, r.api_id);
        sqlite3_bind_int64(apiOpInsert, index++, rowId);
        ret = sqlite3_step(apiOpInsert);
        sqlite3_reset(apiOpInsert);
        ++i;
    }
    tail = tail + i;

    const timestamp_t cb_mid_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
}


void OpTablePrivate::work()
{

}
