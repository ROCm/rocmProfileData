// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <sqlite3.h>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace rlog {

class PropertyDb
{
public:
    // Opens or creates $HOME/.rlog.db. Throws std::runtime_error on failure.
    PropertyDb();
    ~PropertyDb();

    PropertyDb(const PropertyDb&) = delete;
    PropertyDb& operator=(const PropertyDb&) = delete;

    // Looks up (domain, property). If found, returns stored value.
    // If not found, inserts defaultValue and returns it.
    // The returned pointer is stable for the lifetime of this PropertyDb.
    const char* getProperty(const char* domain, const char* property, const char* defaultValue);

private:
    bool openDb(const std::string& path);
    bool enableWal();
    bool initSchema();
    void prepareStatements();
    void finalizeStatements();

    sqlite3*      m_db         = nullptr;
    sqlite3_stmt* m_stmtSelect = nullptr;  // SELECT value FROM properties WHERE domain=? AND property=?
    sqlite3_stmt* m_stmtInsert = nullptr;  // INSERT OR IGNORE INTO properties VALUES(?,?,?)

    std::mutex m_mutex;

    // Cache keyed by (domain, property) for stable const char* lifetime across calls.
    std::map<std::pair<std::string,std::string>, std::string> m_cache;
};

} // namespace rlog
