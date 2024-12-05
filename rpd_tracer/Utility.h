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

#if 0
  // Rainy day module

  #include <amd_comgr/amd_comgr.h>
  #include <cstring>

  amd_comgr_data_t mangled;
  amd_comgr_data_t demangled;
  size_t size;
  char bytes[4096];

  amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled);
  amd_comgr_set_data(mangled, strlen(symbol), symbol);

  amd_comgr_demangle_symbol_name(mangled, &demangled);

  amd_comgr_get_data(demangled, &size, bytes);

  amd_comgr_release_data(mangled);
  amd_comgr_release_data(demangled);
#endif

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

static timestamp_t clocktime_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((timestamp_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);
