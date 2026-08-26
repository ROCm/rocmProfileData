/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <cxxabi.h>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <cstddef>
#include <cstdint>
#include <sqlite3.h>
#include "rlog/client.h"

namespace rpdtracer {

typedef uint64_t timestamp_t;

static inline uint32_t GetPid()
{
    thread_local uint32_t pid = syscall(__NR_getpid);
    return pid;
}

static inline uint32_t GetTid()
{
    thread_local uint32_t tid = syscall(__NR_gettid);
    return tid;
}

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

static timestamp_t clocktime_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((timestamp_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

static inline std::map<std::string, std::string>& configMap()
{
    static std::map<std::string, std::string> s_apiValues;
    return s_apiValues;
}

static inline void setConfig(const char *property, const char *value)
{
    configMap()[property] = value;
}

static inline const char* getConfig(const char *envvar, const char *property, const char *defaultValue)
{
    auto &m = configMap();
    auto it = m.find(property);
    if (it != m.end())
        return it->second.c_str();
    const char *val = std::getenv(envvar);
    if (val != nullptr)
        return val;
    return rlog::getProperty("rpd_tracer", property, defaultValue);
}

static inline int rpdSqliteOpen(const char *basefile, sqlite3 **db)
{
    if (strcmp(basefile, ":memory:") == 0)
        return sqlite3_open_v2("file:rpdmemdb?vfs=memdb", db,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, NULL);
    return sqlite3_open(basefile, db);
}

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);

class Logger;
int unwind(Logger &logger, const char *api, const sqlite_int64 api_id);

}    // namespace rpdtracer
