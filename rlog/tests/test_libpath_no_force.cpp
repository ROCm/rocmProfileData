// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test: RLOG_ROCTX_LIBPATH set but RLOG_FORCE_ROCTX absent.
// roctx should not be enabled or active.

#include <rlog/client.h>

#include <cstdio>
#include <cstdlib>

#ifndef MOCK_ROCTX_PATH
#error "MOCK_ROCTX_PATH must be defined at compile time"
#endif

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
    printf("\n[RLOG_ROCTX_LIBPATH without RLOG_FORCE_ROCTX]\n");

    // RLOG_FORCE_ROCTX is intentionally not set.
    setenv("RLOG_ROCTX_LIBPATH", MOCK_ROCTX_PATH, 1);

    rlog::init();

    // Without RLOG_FORCE_ROCTX, roctx must not be enabled.
    CHECK(!rlog::enabled(rlog::Api::Roctx));

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
