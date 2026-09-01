// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Range benchmark: measures rlog::Range with expensive arguments.
//
// Every variant below formats the same five-field string. The only difference
// is WHEN that formatting happens:
//
//   eager    — the string is built at the call site and passed to the
//              constructor. Arguments are evaluated before the constructor is
//              entered, so the guard inside it is too late to help. The cost is
//              paid on every call whether a tool is attached or not.
//
//   deferred — a lambda is passed instead. The constructor invokes it only
//              after the guard passes, so with no tool attached the formatting
//              never runs at all.
//
// Phase 1 runs with no Logger registered (the production case: instrumented
// application, no profiler attached). Phase 2 registers a no-op Logger so both
// forms are actually logging, showing that the lambda costs nothing extra once
// the work has to be done anyway.

#include <rlog/client.h>
#include <rlog/Logger.h>
#include <rlog/Range.h>
#include "Hub.h"

#include <atomic>
#include <chrono>
#include <cstdio>

static std::atomic<bool> isLogging{false};

static void onActiveChanged()
{
    isLogging.store(rlog::isActive(), std::memory_order_relaxed);
}

class NullLogger : public rlog::Logger {
public:
    void mark(const char*, const char*, const char*, const char*) override {}
    void rangePush(const char*, const char*, const char*, const char*) override {}
    void rangePop() override {}
};

static const int N = 1'000'000;

static volatile int sink = 0;

typedef std::chrono::steady_clock clk;

static double run_empty()
{
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        sink = i;
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_literal()
{
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        rlog::Range r(isLogging, "bench", "range", "op", "static args");
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_eager()
{
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        rlog::Range r(isLogging, "bench", "range", "op",
                      rlog::fmt("m=%d n=%d k=%d alpha=%f beta=%f",
                                i, i + 1, i + 2, 1.0, 0.0).c_str());
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_deferred()
{
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        rlog::Range r(isLogging, "bench", "range", "op",
                      [&]{ return rlog::fmt("m=%d n=%d k=%d alpha=%f beta=%f",
                                            i, i + 1, i + 2, 1.0, 0.0); });
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void report(const char *label, double ms)
{
    printf("  %-28s %8.1f ms  (%7.0f ns/range)\n", label, ms, ms * 1e6 / N);
}

int main()
{
    rlog::init();
    rlog::registerActiveCallback(onActiveChanged);

    printf("range: logging = %s\n\n",
           isLogging.load(std::memory_order_relaxed) ? "true" : "false (baseline)");

    printf("no tool attached:\n");
    report("empty loop", run_empty());
    report("Range, literal args", run_literal());
    report("Range, eager fmt args", run_eager());
    report("Range, deferred lambda", run_deferred());

    NullLogger logger;
    rlog::Hub::singleton().addLogger(logger);

    printf("\nno-op Logger attached (logging = %s):\n",
           isLogging.load(std::memory_order_relaxed) ? "true" : "false");
    report("Range, literal args", run_literal());
    report("Range, eager fmt args", run_eager());
    report("Range, deferred lambda", run_deferred());

    rlog::Hub::singleton().removeLogger(logger);
    return 0;
}
