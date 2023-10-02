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

#include "Utility.h"


//const char *SCHEMA_OP = "CREATE TEMPORARY TABLE \"temp_rocpd_op\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"gpuId\" integer NOT NULL, \"queueId\" integer NOT NULL, \"sequenceId\" integer NOT NULL, \"completionSignal\" varchar(18) NOT NULL, \"start\" integer NOT NULL, \"end\" integer NOT NULL, \"description_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"opType_id\" integer NOT NULL REFERENCES \"rocpd_string\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";

//const char *SCHEMA_API_OPS = "CREATE TEMPORARY TABLE \"temp_rocpd_api_ops\" (\"id\" integer NOT NULL PRIMARY KEY AUTOINCREMENT, \"api_id\" integer NOT NULL REFERENCES \"rocpd_api\" (\"id\") DEFERRABLE INITIALLY DEFERRED, \"op_id\" integer NOT NULL REFERENCES \"rocpd_op\" (\"id\") DEFERRABLE INITIALLY DEFERRED)";


class MetadataTablePrivate
{
public:
    MetadataTablePrivate(MetadataTable *cls) : p(cls) {} 

    sqlite3_stmt *sessionInsert;

    sqlite3_int64 sessionId;
    void createSession();

    MetadataTable *p;
};

int sessionCallback(void *data, int argc, char **argv, char **colName)
{
    sqlite3_int64 &sessionId = *(sqlite3_int64*)data;
    sessionId = atoll(argv[0]);
    return 0;
}

MetadataTable::MetadataTable(const char *basefile)
: Table(basefile)
, d(new MetadataTablePrivate(this))
{
    d->createSession();
}

void MetadataTable::flush()
{
}

void MetadataTable::finalize()
{
}

sqlite3_int64 MetadataTable::sessionId()
{
	return d->sessionId;
}


void MetadataTablePrivate::createSession()
{
    int ret;
    sqlite3_exec(p->m_connection, "BEGIN EXCLUSIVE TRANSACTION", NULL, NULL, NULL);
    // get or create session count property

    sqlite3_int64 sessionId = -1;
    char *error_msg;
    ret = sqlite3_exec(p->m_connection, "SELECT value FROM rocpd_metadata WHERE tag = 'session_count'", &sessionCallback, &sessionId, &error_msg);
    if (sessionId == -1) {
        sessionId = 0;
        ret = sqlite3_exec(p->m_connection, "INSERT into rocpd_metadata(tag, value) VALUES ('session_count', 1)", NULL, NULL, &error_msg);
    }
    else {
        char buff[4096];
        std::snprintf(buff, 4096, "UPDATE rocpd_metadata SET value = '%lld' WHERE tag = 'session_count'", sessionId + 1);
        ret = sqlite3_exec(p->m_connection, buff, NULL, NULL, &error_msg);
    }

    sqlite3_exec(p->m_connection, "END TRANSACTION", NULL, NULL, NULL);

    //printf("Opening session: %lld\n", sessionId);
    fflush(stdout);

    this->sessionId = sessionId;
}
