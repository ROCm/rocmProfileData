#include "Table.h"
#include <thread>

#include "hsa_rsrc_factory.h"

typedef uint64_t timestamp_t;


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
    fflush(stdout);
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
