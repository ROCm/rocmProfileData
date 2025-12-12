/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <cxxabi.h>
#include <string>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <time.h>      // For clock_gettime and CLOCK_MONOTONIC
#include <sqlite3.h>   // For sqlite_int64


typedef uint64_t timestamp_t;

// Declare the namespace and its atomic variables as extern
namespace chrono_sync {
    extern std::atomic<int64_t> offset;
    extern std::atomic<int64_t> drift;
    extern std::atomic<int64_t> last_host_timestamp;
}

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

// majority of the timestamps are recorded here
static timestamp_t clocktime_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // return ((timestamp_t)ts.tv_sec * 1000000000) + ts.tv_nsec; // TODO FOR ALI: ADD the offset here
    // printf("Offset in clocktime_ns: %ld\n", chrono_sync::offset.load());
    // Add the offset to the returned timestamp offset would be chrono_sync::offset + drift * elapsed_time_since_last_sync
    timestamp_t now_ns = ((timestamp_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
    timestamp_t elapsed_since_last_sync = now_ns - chrono_sync::last_host_timestamp.load();
    timestamp_t offset =  chrono_sync::offset.load() + chrono_sync::drift.load() * elapsed_since_last_sync;
    // print all the values
    printf("now_ns is %ld, last_host_timestamp is %ld\n", now_ns, chrono_sync::last_host_timestamp.load());
    // print now_ns + offset
    printf("Adjusted time is %ld\n", now_ns + offset);
    // print chrono_sync::offset and chrono_sync::drift
    printf("chrono_sync::offset is %ld and chrono_sync::drift offset is %ld\n", chrono_sync::offset.load(), chrono_sync::drift.load() * elapsed_since_last_sync);
    printf("offset is %ld and lt is %ld\n", offset, elapsed_since_last_sync);
    return now_ns + offset ; // TODO FOR ALI: ADD the offset here
}

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);

void update_clock_offset(int64_t new_offset);

void adjust_offset_by_delta(int64_t delta);

void update_drift(int64_t new_drift);

int64_t get_current_offset();

// Example: Update all sync parameters atomically
void sync_clock(int64_t measured_offset, int64_t measured_drift);

class Logger;
int unwind(Logger &logger, const char *api, const sqlite_int64 api_id);
