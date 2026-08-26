/*
 * Benchmark: time rlog::mark() calls with strings from a sample RPD file.
 * Run under runTracer.sh to profile with librpd_tracer.so.
 *
 * Usage: runTracer.sh ./bench_rlog_strings [sample.rpd]
 *        Default sample: /data/test/sample_rlog_resnet50.rpd
 */
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <sqlite3.h>

#include "rlog/client.h"

struct Row {
    std::string domain;
    std::string category;
    std::string apiName;
};

static std::vector<Row> loadRows(const char *dbpath)
{
    std::vector<Row> rows;
    sqlite3 *db = nullptr;
    if (sqlite3_open_v2(dbpath, &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        fprintf(stderr, "Cannot open %s: %s\n", dbpath, sqlite3_errmsg(db));
        return rows;
    }

    const char *sql =
        "SELECT d.string, c.string, n.string "
        "FROM rocpd_api a "
        "JOIN rocpd_string d ON a.domain_id = d.id "
        "JOIN rocpd_string c ON a.category_id = c.id "
        "JOIN rocpd_string n ON a.apiName_id = n.id "
        "WHERE d.string IN ('torch', 'miopen')";

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return rows;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Row r;
        r.domain = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.apiName = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        rows.push_back(r);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return rows;
}

int main(int argc, char **argv)
{
    const char *dbpath = argc > 1 ? argv[1] : "/data/test/sample_rlog_resnet50.rpd";

    fprintf(stderr, "Loading rows from %s...\n", dbpath);
    auto rows = loadRows(dbpath);
    if (rows.empty()) {
        fprintf(stderr, "No rows loaded\n");
        return 1;
    }
    fprintf(stderr, "Loaded %zu rlog rows\n", rows.size());

    rlog::init();
    fprintf(stderr, "rlog active: %d\n", rlog::isActive());

    // Warm up
    for (size_t i = 0; i < 1000 && i < rows.size(); ++i)
        rlog::mark(rows[i].domain.c_str(), rows[i].category.c_str(), rows[i].apiName.c_str(), "");
    fprintf(stderr, "Warmup done\n");

    const int PASSES = 5;
    for (int pass = 0; pass < PASSES; ++pass) {
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < rows.size(); ++i)
            rlog::mark("torch", "function", "aten::empty", "");

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double rate = rows.size() / (ms / 1000.0);
        fprintf(stderr, "Pass %d: %zu marks in %.1f ms  (%.0f marks/sec)\n",
                pass + 1, rows.size(), ms, rate);
    }

    fprintf(stderr, "Done\n");
    return 0;
}
