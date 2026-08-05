// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mock roctx shared library for testing.
// Implements the roctx symbols that rlog.cpp loads via dlopen.
// Exports counters so tests can verify calls were made.

#include <cstring>

extern "C" {

int mock_roctx_mark_count    = 0;
int mock_roctx_push_count    = 0;
int mock_roctx_pop_count     = 0;

void roctxMarkA(const char* /*message*/)
{
    ++mock_roctx_mark_count;
}

void roctxRangePushA(const char* /*message*/)
{
    ++mock_roctx_push_count;
}

void roctxRangePop()
{
    ++mock_roctx_pop_count;
}

} // extern "C"
