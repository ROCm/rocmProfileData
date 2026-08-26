/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Table.h"
#include "Utility.h"
#include <cstring>

using rpdtracer::Table;

int busy_handler(void *data, int count)
{
    count = (count < 9) ? count : 8;
    usleep(1000 * (0x1 << count));
    return 1;
}

static int wal_check_callback(void *data, int ncols, char **values, char **names)
{
    if (ncols > 0 && values[0])
        *static_cast<bool*>(data) = (strcmp(values[0], "wal") == 0);
    return 0;
}

Table::Table(const char *basefile)
: m_connection(NULL)
{
    rpdSqliteOpen(basefile, &m_connection);
    sqlite3_busy_handler(m_connection, &busy_handler, NULL);

    bool walEnabled = false;
    sqlite3_exec(m_connection, "PRAGMA journal_mode=WAL", wal_check_callback, &walEnabled, NULL);
    if (walEnabled)
        sqlite3_exec(m_connection, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);
}

Table::~Table()
{
    // FIXME: ensure these aren't in use
    //pthread_mutex_destroy(m_mutex);
    //pthread_cond_destroy(m_wait);

    sqlite3_close(m_connection);
}

void Table::setIdOffset(sqlite3_int64 offset)
{
    m_idOffset = offset;
}
