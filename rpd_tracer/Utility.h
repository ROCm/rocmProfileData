/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

//FIXME make a universal timestamp source
//static timestamp_t clocktime_ns() { return 0; }

#include <cupti.h>
static timestamp_t clocktime_ns() {
    timestamp_t ts;
    cuptiGetTimestamp(&ts);
    return ts;
}

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);
