// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for RLOG_FORCE_ROCTX and RLOG_ROCTX_LIBPATH environment variables.
//
// Each test sets env vars, calls rlog::init(), exercises the API, then reads
// call counters from mock_roctx.so via dlsym to verify dispatch happened.
//
// The path to mock_roctx.so is injected at build time via MOCK_ROCTX_PATH.

#include <rlog/client.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#ifndef MOCK_ROCTX_PATH
#error "MOCK_ROCTX_PATH must be defined at compile time"
#endif

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

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

static void begin(const char* name)
{
    printf("\n[%s]\n", name);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Open mock_roctx.so and return a pointer to one of its int counters.
// Returns nullptr if the symbol is not found.
static int* counter(void* handle, const char* name)
{
    return static_cast<int*>(dlsym(handle, name));
}

// Re-initialise global state between tests by clearing env vars and
// reloading rlog.cpp state.  Since rlog.cpp uses static globals we
// can't truly reset them between tests in a single process, so each
// test case is designed to be run independently via ctest (one binary
// per test invocation via separate test commands that set env vars).
// Within a single run we can still exercise multiple checks per test.

// ---------------------------------------------------------------------------
// Test: RLOG_FORCE_ROCTX + RLOG_ROCTX_LIBPATH
// ---------------------------------------------------------------------------

static int test_force_roctx()
{
    begin("RLOG_FORCE_ROCTX with RLOG_ROCTX_LIBPATH");

    setenv("RLOG_FORCE_ROCTX",    "1",              1);
    setenv("RLOG_ROCTX_LIBPATH",  MOCK_ROCTX_PATH, 1);

    rlog::init();

    // After init with RLOG_FORCE_ROCTX=1, isActive() must return true
    CHECK(rlog::isActive());
    CHECK(rlog::enabled(rlog::Api::Roctx));

    // Open the same mock so we can read its counters (RTLD_NOLOAD shares
    // the already-loaded instance that rlog opened via dlopen)
    void* handle = dlopen(MOCK_ROCTX_PATH, RTLD_LAZY | RTLD_NOLOAD);
    CHECK(handle != nullptr);
    if (!handle) return 1;

    int* mark_count = counter(handle, "mock_roctx_mark_count");
    int* push_count = counter(handle, "mock_roctx_push_count");
    int* pop_count  = counter(handle, "mock_roctx_pop_count");
    CHECK(mark_count != nullptr);
    CHECK(push_count != nullptr);
    CHECK(pop_count  != nullptr);
    if (!mark_count || !push_count || !pop_count) {
        dlclose(handle);
        return 1;
    }

    int before_mark = *mark_count;
    int before_push = *push_count;
    int before_pop  = *pop_count;

    rlog::mark("domain", "cat", "api", "args");
    CHECK(*mark_count == before_mark + 1);

    rlog::rangePush("domain", "cat", "api", "args");
    CHECK(*push_count == before_push + 1);

    rlog::rangePop();
    CHECK(*pop_count == before_pop + 1);

    dlclose(handle);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    test_force_roctx();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
