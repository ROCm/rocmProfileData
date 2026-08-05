// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <sqlite3.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static std::string dbPath()
{
    const char* home = getenv("HOME");
    return (home ? std::string(home) : std::string(".")) + "/.rlog.db";
}

static sqlite3* openDb(const std::string& path, bool create)
{
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX;
    if (create)
        flags |= SQLITE_OPEN_CREATE;

    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(path.c_str(), &db, flags, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "rlog-config: cannot open " << path;
        if (db) std::cerr << ": " << sqlite3_errmsg(db);
        std::cerr << "\n";
        if (db) sqlite3_close(db);
        return nullptr;
    }
    sqlite3_busy_timeout(db, 5000);

    if (create) {
        char* errmsg = nullptr;
        sqlite3_exec(db,
            "PRAGMA journal_mode=WAL;"
            "CREATE TABLE IF NOT EXISTS properties ("
            "  domain   TEXT NOT NULL,"
            "  property TEXT NOT NULL,"
            "  value    TEXT NOT NULL,"
            "  PRIMARY KEY (domain, property)"
            ");",
            nullptr, nullptr, &errmsg);
        if (errmsg) { sqlite3_free(errmsg); }
    }
    return db;
}

static int cmdList(sqlite3* db)
{
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db,
        "SELECT domain, property, value FROM properties ORDER BY domain, property;",
        -1, &stmt, nullptr);

    std::string currentDomain;
    bool any = false;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        any = true;
        const char* domain   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* property = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* value    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        if (currentDomain != domain) {
            if (!currentDomain.empty())
                std::cout << "\n";
            std::cout << "[" << domain << "]\n";
            currentDomain = domain;
        }
        std::cout << "  " << property << " = " << value << "\n";
    }
    (void)any;
    sqlite3_finalize(stmt);
    return 0;
}

static int cmdGet(sqlite3* db, const std::string& domain, const std::string& property)
{
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db,
        "SELECT value FROM properties WHERE domain=? AND property=?;",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, domain.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, property.c_str(), -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const char* value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::cout << (value ? value : "") << "\n";
        sqlite3_finalize(stmt);
        return 0;
    }

    sqlite3_finalize(stmt);
    std::cerr << "rlog-config: property not found: " << domain << ":" << property << "\n";
    return 1;
}

static int cmdSet(sqlite3* db, const std::string& domain, const std::string& property, const std::string& value)
{
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO properties(domain, property, value) VALUES(?,?,?)"
        "  ON CONFLICT(domain, property) DO UPDATE SET value=excluded.value;",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, domain.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, property.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, value.c_str(),    -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "rlog-config: set failed: " << sqlite3_errmsg(db) << "\n";
        return 1;
    }
    return 0;
}

static void usage()
{
    std::cerr <<
        "Usage:\n"
        "  rlog-config                          List all properties grouped by domain\n"
        "  rlog-config get <domain>:<property>  Print the value of a property\n"
        "  rlog-config set <domain>:<property> <value>  Set a property value\n";
}

static bool parseDomainProperty(const char* arg, std::string& domain, std::string& property)
{
    const char* colon = strchr(arg, ':');
    if (!colon || colon == arg || *(colon + 1) == '\0') {
        std::cerr << "rlog-config: expected <domain>:<property>, got: " << arg << "\n";
        return false;
    }
    domain   = std::string(arg, colon);
    property = std::string(colon + 1);
    return true;
}

int main(int argc, char** argv)
{
    const std::string path = dbPath();

    if (argc == 1) {
        sqlite3* db = openDb(path, false);
        if (!db) return 1;
        int rc = cmdList(db);
        sqlite3_close(db);
        return rc;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "get") == 0) {
        if (argc != 3) { usage(); return 1; }
        std::string domain, property;
        if (!parseDomainProperty(argv[2], domain, property)) return 1;
        sqlite3* db = openDb(path, false);
        if (!db) return 1;
        int rc = cmdGet(db, domain, property);
        sqlite3_close(db);
        return rc;
    }

    if (strcmp(cmd, "set") == 0) {
        if (argc != 4) { usage(); return 1; }
        std::string domain, property;
        if (!parseDomainProperty(argv[2], domain, property)) return 1;
        sqlite3* db = openDb(path, true);
        if (!db) return 1;
        int rc = cmdSet(db, domain, property, argv[3]);
        sqlite3_close(db);
        return rc;
    }

    std::cerr << "rlog-config: unknown command: " << cmd << "\n";
    usage();
    return 1;
}
