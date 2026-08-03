// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// roctx benchmark: same sustained + burst measurement as bench_external but
// dispatches through roctx (libroctx64.so) instead of the rlog Hub.
// rlog is explicitly disabled so only the roctx path is exercised.
//
// Skips with a message if libroctx64.so is not installed.
//
// Run under a tool to measure roctx service throughput:
//
//   loadTracer.sh ./bench_roctx
//   rocprofv3 --sys-trace ./bench_roctx

#include <rlog/client.h>

#include <chrono>
#include <cstdio>

int main()
{
    rlog::init();
    rlog::setEnabled(rlog::Api::Rlog,  false);
    rlog::setEnabled(rlog::Api::Roctx, true);

    if (!rlog::enabled(rlog::Api::Roctx)) {
        fprintf(stderr, "roctx not active — is libroctx64.so installed?\n");
        return 0;
    }

    printf("roctx: tool active = %s\n\n", rlog::isActive() ? "yes" : "no (baseline)");

    // --- sustained + burst window ---
    {
        const int N          = 1'000'000;
        const int WARMUP     = 1000;
        const int BURST_SIZE = 10000;

        auto t_sustained_0 = std::chrono::steady_clock::now();

        for (int i = 0; i < WARMUP; ++i) {
            rlog::rangePush("bench", "roctx", "op", "");
            rlog::rangePop();
        }

        auto t_burst_0 = std::chrono::steady_clock::now();
        for (int i = 0; i < BURST_SIZE; ++i) {
            rlog::rangePush("bench", "roctx", "op", "");
            rlog::rangePop();
        }
        auto t_burst_1 = std::chrono::steady_clock::now();

        for (int i = WARMUP + BURST_SIZE; i < N; ++i) {
            rlog::rangePush("bench", "roctx", "op", "");
            rlog::rangePop();
        }

        auto t_sustained_1 = std::chrono::steady_clock::now();

        double burst_ms = std::chrono::duration<double, std::milli>(t_burst_1 - t_burst_0).count();
        double sust_ms  = std::chrono::duration<double, std::milli>(t_sustained_1 - t_sustained_0).count();

        printf("burst:     %7d pairs  %8.1f ms  (%5.0f ns/pair  %.2f Mpairs/sec)"
               "  [after %d warmup]\n",
               BURST_SIZE, burst_ms, burst_ms * 1e6 / BURST_SIZE,
               BURST_SIZE / (burst_ms * 1e3), WARMUP);
        printf("sustained: %7d pairs  %8.1f ms  (%5.0f ns/pair  %.2f Mpairs/sec)\n",
               N, sust_ms, sust_ms * 1e6 / N, N / (sust_ms * 1e3));
    }

    return 0;
}
