// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Table.h"

#include <thread>

#include "Utility.h"

using rpdtracer::MetadataTable;

namespace rpdtracer {

class MetadataTablePrivate
{
public:
    MetadataTablePrivate(MetadataTable *cls) : p(cls) {} 

    sqlite3_stmt *sessionInsert;
    sqlite3_stmt *metaInsert;

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
    sqlite3_prepare_v2(m_connection, "INSERT INTO rocpd_metadata(tag, value) VALUES (?,?)", -1, &d->metaInsert, NULL);
    d->createSession();
}

void MetadataTable::flush()
{
}

void MetadataTable::finalize()
{
}

void MetadataTable::insert(const std::string &tag, const std::string &value)
{
    sqlite3_exec(m_connection, "BEGIN", NULL, NULL, NULL);
    sqlite3_bind_text(d->metaInsert, 1, tag.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(d->metaInsert, 2, value.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(d->metaInsert);
    sqlite3_reset(d->metaInsert);
    sqlite3_exec(m_connection, "END", NULL, NULL, NULL);
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

}  // namespace rpdtracer
