// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test: RLOG_FORCE_NVTX with the real libcupti.so at its default path.
// Skips (exit 77) if the library is not installed.
// Verifies: library loads, nvtx becomes active, mark/push/pop don't crash.

#include <rlog/client.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#define NVTX_LIB "libcupti.so"
#define SKIP_CODE 77

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* expr, const char* file, int line)
{
    if (cond) {
        printf("  PASS: %s\n", expr);
        ++g_pass;
    } else {
        printf("  FAIL: %s  (%s:%d)\n", expr, file, line);
        ++g_fail;
    }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

int main()
{
    printf("\n[RLOG_FORCE_NVTX with real %s]\n", NVTX_LIB);

    // Probe: can the real library be found?
    void* probe = dlopen(NVTX_LIB, RTLD_LAZY);
    if (!probe) {
        printf("  SKIP: %s not found (%s)\n", NVTX_LIB, dlerror());
        return SKIP_CODE;
    }
    dlclose(probe);

    setenv("RLOG_FORCE_NVTX", "1", 1);
    // RLOG_NVTX_LIBPATH not set — init() uses default "libcupti.so"

    rlog::init();

    CHECK(rlog::enabled(rlog::Api::Nvtx));
    CHECK(rlog::isActive());

    // Exercise the API — must not crash
    rlog::mark("domain", "cat", "api", "args");
    rlog::rangePush("domain", "cat", "api", "args");
    rlog::rangePop();
    printf("  PASS: mark/rangePush/rangePop did not crash\n");
    ++g_pass;

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
