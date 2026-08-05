// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "PropertyDb.h"

#include <cstdlib>
#include <stdexcept>

namespace rlog {

PropertyDb::PropertyDb()
{
    const char* home = getenv("HOME");
    std::string path = (home ? std::string(home) : std::string(".")) + "/.rlog.db";

    if (!openDb(path) || !enableWal() || !initSchema()) {
        std::string msg = "PropertyDb: failed to open ";
        msg += path;
        if (m_db) {
            msg += ": ";
            msg += sqlite3_errmsg(m_db);
            sqlite3_close(m_db);
            m_db = nullptr;
        }
        throw std::runtime_error(msg);
    }
    prepareStatements();
}

PropertyDb::~PropertyDb()
{
    finalizeStatements();
    if (m_db) {
        sqlite3_close(m_db);
        m_db = nullptr;
    }
}

bool PropertyDb::openDb(const std::string& path)
{
    int rc = sqlite3_open_v2(
        path.c_str(),
        &m_db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        nullptr);
    if (rc != SQLITE_OK) {
        if (m_db) { sqlite3_close(m_db); m_db = nullptr; }
        return false;
    }
    sqlite3_busy_timeout(m_db, 5000);
    return true;
}

bool PropertyDb::enableWal()
{
    char* errmsg = nullptr;
    int rc = sqlite3_exec(m_db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &errmsg);
    if (errmsg) sqlite3_free(errmsg);
    return rc == SQLITE_OK;
}

bool PropertyDb::initSchema()
{
    const char* sql =
        "CREATE TABLE IF NOT EXISTS properties ("
        "  domain   TEXT NOT NULL,"
        "  property TEXT NOT NULL,"
        "  value    TEXT NOT NULL,"
        "  PRIMARY KEY (domain, property)"
        ");";
    char* errmsg = nullptr;
    int rc = sqlite3_exec(m_db, sql, nullptr, nullptr, &errmsg);
    if (errmsg) sqlite3_free(errmsg);
    return rc == SQLITE_OK;
}

void PropertyDb::prepareStatements()
{
    sqlite3_prepare_v2(m_db,
        "SELECT value FROM properties WHERE domain=? AND property=?;",
        -1, &m_stmtSelect, nullptr);
    sqlite3_prepare_v2(m_db,
        "INSERT OR IGNORE INTO properties(domain, property, value) VALUES(?,?,?);",
        -1, &m_stmtInsert, nullptr);
}

void PropertyDb::finalizeStatements()
{
    if (m_stmtSelect) { sqlite3_finalize(m_stmtSelect); m_stmtSelect = nullptr; }
    if (m_stmtInsert) { sqlite3_finalize(m_stmtInsert); m_stmtInsert = nullptr; }
}

const char* PropertyDb::getProperty(const char* domain, const char* property, const char* defaultValue)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto key = std::make_pair(std::string(domain), std::string(property));

    auto it = m_cache.find(key);
    if (it != m_cache.end())
        return it->second.c_str();

    if (!m_db || !m_stmtSelect || !m_stmtInsert)
        return defaultValue;

    // SELECT existing row
    sqlite3_reset(m_stmtSelect);
    sqlite3_bind_text(m_stmtSelect, 1, domain,   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(m_stmtSelect, 2, property, -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(m_stmtSelect);
    if (rc == SQLITE_ROW) {
        const char* val = reinterpret_cast<const char*>(sqlite3_column_text(m_stmtSelect, 0));
        m_cache[key] = val ? val : "";
        sqlite3_reset(m_stmtSelect);
        return m_cache[key].c_str();
    }
    sqlite3_reset(m_stmtSelect);

    // Row not found — insert defaultValue
    const char* insertVal = defaultValue ? defaultValue : "";

    char* errmsg = nullptr;
    sqlite3_exec(m_db, "BEGIN IMMEDIATE;", nullptr, nullptr, &errmsg);
    if (errmsg) { sqlite3_free(errmsg); errmsg = nullptr; }

    sqlite3_reset(m_stmtInsert);
    sqlite3_bind_text(m_stmtInsert, 1, domain,    -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(m_stmtInsert, 2, property,  -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(m_stmtInsert, 3, insertVal, -1, SQLITE_TRANSIENT);
    sqlite3_step(m_stmtInsert);
    sqlite3_reset(m_stmtInsert);

    sqlite3_exec(m_db, "COMMIT;", nullptr, nullptr, &errmsg);
    if (errmsg) { sqlite3_free(errmsg); }

    m_cache[key] = insertVal;
    return m_cache[key].c_str();
}

} // namespace rlog
