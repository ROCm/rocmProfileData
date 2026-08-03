// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loopback benchmark: measures rlog internal dispatch overhead.
// A no-op Logger is registered so calls traverse the full client→Hub path
// but do no real work. Run standalone — no external tool required.

#include <rlog/client.h>
#include <rlog/Logger.h>
#include "Hub.h"

#include <chrono>
#include <cstdio>

class NullLogger : public rlog::Logger {
public:
    void mark(const char*, const char*, const char*, const char*) override {}
    void rangePush(const char*, const char*, const char*, const char*) override {}
    void rangePop() override {}
};

int main()
{
    NullLogger logger;
    rlog::Hub::singleton().addLogger(logger);
    rlog::init();

    const int N = 1'000'000;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        rlog::rangePush("bench", "loopback", "op", "");
        rlog::rangePop();
    }
    auto t1 = std::chrono::steady_clock::now();

    double ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_pair = ms * 1e6 / N;
    double mpairs  = N / (ms * 1e3);

    printf("loopback: %d iterations  %.1f ms  (%.0f ns/pair  %.2f Mpairs/sec)\n",
           N, ms, ns_pair, mpairs);

    rlog::Hub::singleton().removeLogger(logger);
    return 0;
}
