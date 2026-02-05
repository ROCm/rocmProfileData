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

#include <map>
#include <thread>
#include <array>
#include <mutex>

#include "rpd_tracer.h"
#include "Utility.h"


const char *SCHEMA_OP = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_op" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
)|";

const char *SCHEMA_API_OPS = R"|(
CREATE TEMPORARY TABLE "temp_rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" bigint NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" bigint NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
)|";


class OpTablePrivate
{
public:
    OpTablePrivate(OpTable *cls) : p(cls) {} 
    static const int BUFFERSIZE = 4096 * 4;
    static const int BATCHSIZE = 4096;           // rows per transaction
    std::array<OpTable::row, BUFFERSIZE> rows; // Circular buffer
    std::map<sqlite3_int64, sqlite3_int64> descriptions;
    std::mutex descriptionLock;

    sqlite3_stmt *opInsert;
    sqlite3_stmt *apiOpInsert;

    OpTable *p;
};


OpTable::OpTable(const char *basefile)
: BufferedTable(basefile, OpTablePrivate::BUFFERSIZE, OpTablePrivate::BATCHSIZE)
, d(new OpTablePrivate(this))
{
    int ret;
    // set up tmp tables
    ret = sqlite3_exec(m_connection, SCHEMA_OP, NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, SCHEMA_API_OPS, NULL, NULL, NULL);

    // prepare queries to insert row
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_op(id, gpuId, queueId, sequenceId, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?)", -1, &d->opInsert, NULL);
    ret = sqlite3_prepare_v2(m_connection, "insert into temp_rocpd_api_ops(api_id, op_id) values (?,?)", -1, &d->apiOpInsert, NULL);
}


OpTable::~OpTable()
{
    delete d;
}


void OpTable::insert(const OpTable::row &row)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_head - m_tail >= OpTablePrivate::BUFFERSIZE) {
        // buffer is full; insert in-line or wait
        m_wait.notify_one();  // make sure working is running
        m_wait.wait(lock);
    }

    d->rows[(++m_head) % OpTablePrivate::BUFFERSIZE] = row;

    if (workerRunning() == false && (m_head - m_tail) >= OpTablePrivate::BATCHSIZE) {
        //lock.unlock();
        m_wait.notify_one();
    }
}

void OpTable::associateDescription(const sqlite3_int64 &api_id, const sqlite3_int64 &string_id)
{
// Disable this for now.  Getting kernel names from roctracer op records now.
#if 0
    std::lock_guard<std::mutex> guard(d->descriptionLock);
    d->descriptions[api_id] = string_id;
#endif
}

void OpTable::flushRows()
{
    int ret = 0;
    ret = sqlite3_exec(m_connection, "begin transaction", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "insert into rocpd_op select * from temp_rocpd_op", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_op: %d\n", ret);
    ret = sqlite3_exec(m_connection, "insert into rocpd_api_ops (api_id, op_id) select api_id, op_id from temp_rocpd_api_ops", NULL, NULL, NULL);
    fprintf(stderr, "rocpd_api_ops: %d\n", ret);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_op", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "delete from temp_rocpd_api_ops", NULL, NULL, NULL);
    ret = sqlite3_exec(m_connection, "commit", NULL, NULL, NULL);
}


void OpTable::writeRows()
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
        // insert rocpd_op
        int index = 1;
        OpTable::row &r = d->rows[i % BUFFERSIZE];
        sqlite3_int64 primaryKey = i + m_idOffset;

// Disable this for now.  Getting kernel names from roctracer op records now.
#if 0
        // check for description override
        {
            std::lock_guard<std::mutex> guard(d->descriptionLock);
            auto it = d->descriptions.find(r.api_id);
            if (it != d->descriptions.end()) {
                r.description_id = it->second;
                d->descriptions.erase(it);
            }
        }
#endif
        sqlite3_bind_int64(d->opInsert, index++, primaryKey);
        sqlite3_bind_int(d->opInsert, index++, r.gpuId);
        sqlite3_bind_int(d->opInsert, index++, r.queueId);
        sqlite3_bind_int(d->opInsert, index++, r.sequenceId);
        sqlite3_bind_int64(d->opInsert, index++, r.start);
        sqlite3_bind_int64(d->opInsert, index++, r.end);
        sqlite3_bind_int64(d->opInsert, index++, r.description_id + m_idOffset);
        sqlite3_bind_int64(d->opInsert, index++, r.opType_id + m_idOffset);
        int ret = sqlite3_step(d->opInsert);
        sqlite3_reset(d->opInsert);

        // Insert rocpd_api_ops
        //sqlite_int64 rowId = sqlite3_last_insert_rowid(m_connection);
        index = 1;
        sqlite3_bind_int64(d->apiOpInsert, index++, sqlite3_int64(r.api_id) + m_idOffset);
        sqlite3_bind_int64(d->apiOpInsert, index++, sqlite3_int64(i) + m_idOffset);
        ret = sqlite3_step(d->apiOpInsert);
        sqlite3_reset(d->apiOpInsert);
    }
    lock.lock();
    m_tail = end;
    lock.unlock();

    sqlite3_exec(m_connection, "END TRANSACTION", NULL, NULL, NULL);
    const timestamp_t cb_end_time = clocktime_ns() + 1;
    char buff[4096];
    std::snprintf(buff, 4096, "count=%d | remaining=%d", end - start + 1, m_head - m_tail);
    createOverheadRecord(cb_begin_time, cb_end_time, "OpTable::writeRows", buff);
}
