/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <cstdarg>
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
#include <atomic>
#include <time.h>
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

// Seqlock-protected clock sync state, written by ChronoSync's Firefly algorithm.
// Inlined here so the seq==0 fast path (no sync active) costs only a single
// load+branch on the ~80 clocktime_ns() call sites in the tracing hot path.
namespace firefly {

struct SvcState {
    std::atomic<unsigned int> sequence{0};
    timespec referenceTime{};
    int64_t  offset{0};
    double   drift{0.0};
};

extern SvcState* g_pSvcState;

void create_clocksync_shm(std::string& shm_name_out);
void attach_clocksync_shm(const std::string& shm_name);
void cleanup_clocksync_shm(const std::string& shm_name);

} // namespace firefly

static inline int64_t svc_read_offset_ns(timestamp_t now_ns) {
    unsigned seq = firefly::g_pSvcState->sequence.load(std::memory_order_acquire);
    if (seq == 0)
        return 0;

    int64_t offset;
    double drift;
    timestamp_t ref_ns;
    do {
        seq = firefly::g_pSvcState->sequence.load(std::memory_order_acquire);
        offset = firefly::g_pSvcState->offset;
        drift = firefly::g_pSvcState->drift;
        ref_ns = timespec_to_ns(firefly::g_pSvcState->referenceTime);
    } while ((seq & 1) || firefly::g_pSvcState->sequence.load(std::memory_order_acquire) != seq);

    int64_t elapsed = static_cast<int64_t>(now_ns - ref_ns);
    return offset + static_cast<int64_t>(drift * elapsed);
}

static inline timestamp_t adjust_external_ts(timestamp_t ts) {
    return ts + svc_read_offset_ns(ts);
}

static inline timestamp_t clocktime_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timestamp_t now_ns = ((timestamp_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
    return now_ns + svc_read_offset_ns(now_ns);
}

std::map<std::string, std::string>& configMap();
void setConfig(const char *property, const char *value);
const char* getConfig(const char *envvar, const char *property, const char *defaultValue);

static inline bool rpdQuiet()
{
    static bool quiet = (atoi(getConfig("RPDT_QUIET", "quiet", "0")) != 0);
    return quiet;
}

__attribute__((format(printf, 1, 2)))
static inline void rpdLog(const char *fmt, ...)
{
    if (rpdQuiet())
        return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static inline int rpdSqliteOpen(const char *basefile, sqlite3 **db)
{
    if (strcmp(basefile, ":memory:") == 0)
        return sqlite3_open_v2("file:rpdmemdb?vfs=memdb", db,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, NULL);
    return sqlite3_open(basefile, db);
}

int sqlite_busy_handler(void *data, int count);

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);

class Logger;
int unwind(Logger &logger, const char *api, const sqlite_int64 api_id);

}    // namespace rpdtracer
