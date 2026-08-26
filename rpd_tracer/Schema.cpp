/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Schema.h"

#include <sqlite3.h>
#include "Utility.h"
#include <stdio.h>
#include <string.h>

#include "tableSchema.h"
#include "utilitySchema.h"

namespace rpdtracer {

void ensureSchema(const char *basefile)
{
    sqlite3 *db = nullptr;
    int ret = rpdSqliteOpen(basefile, &db);
    if (ret != SQLITE_OK) {
        fprintf(stderr, "rpd_tracer: cannot open database %s: %s\n", basefile, sqlite3_errmsg(db));
        return;
    }

    // Use BEGIN EXCLUSIVE to serialize against other processes
    ret = sqlite3_exec(db, "BEGIN EXCLUSIVE TRANSACTION", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
        fprintf(stderr, "rpd_tracer: cannot lock database %s: %s\n", basefile, sqlite3_errmsg(db));
        sqlite3_close(db);
        return;
    }

    // Check if schema already exists
    sqlite3_stmt *stmt = nullptr;
    bool hasSchema = false;
    ret = sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rocpd_string'", -1, &stmt, nullptr);
    if (ret == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW)
            hasSchema = true;
        sqlite3_finalize(stmt);
    }

    if (!hasSchema) {
        // tableSchema and utilitySchema are null-terminated via xxd -i
        char *errmsg = nullptr;
        ret = sqlite3_exec(db, reinterpret_cast<const char *>(tableSchema_cmd), nullptr, nullptr, &errmsg);
        if (ret != SQLITE_OK) {
            fprintf(stderr, "rpd_tracer: tableSchema error: %s\n", errmsg);
            sqlite3_free(errmsg);
        }

        ret = sqlite3_exec(db, reinterpret_cast<const char *>(utilitySchema_cmd), nullptr, nullptr, &errmsg);
        if (ret != SQLITE_OK) {
            fprintf(stderr, "rpd_tracer: utilitySchema error: %s\n", errmsg);
            sqlite3_free(errmsg);
        }
    }

    sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);
    sqlite3_close(db);
}

}  // namespace rpdtracer
