/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <cxxabi.h>

typedef uint64_t timestamp_t;
static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

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

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);
