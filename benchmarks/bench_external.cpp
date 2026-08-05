// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// External tool benchmark: measures end-to-end throughput when a profiling
// tool is attached. No Logger is registered by this binary — the tool
// (rpd_tracer or rocprofv3) registers its own Logger with the Hub at launch.
//
// Run standalone to get a no-tool baseline, then compare:
//
//   loadTracer.sh ./bench_external
//   rocprofv3 --sys-trace ./bench_external
//
// Two tests are run:
//
//   sustained — 1M pairs back-to-back. Stresses steady-state throughput and
//               exposes tools that fall behind when their buffers fill.
//
//   burst     — 1000 pairs, then 10 ms rest, repeated 100 times (100K pairs
//               total). The rest gives tools time to flush between bursts, so
//               each burst hits a drained buffer. A tool with good burst
//               performance but poor sustained performance likely relies on
//               buffering and cannot keep up under continuous load.

#include <rlog/client.h>

#include <chrono>
#include <cstdio>

int main()
{
    rlog::init();

    printf("external: tool active = %s\n\n", rlog::isActive() ? "yes" : "no (baseline)");

    // --- sustained + burst window ---
    // The burst time is captured from within the sustained run:
    //   - 1000 pairs warmup (unmetered)
    //   - 10000 pairs timed as the burst window
    //   - remainder timed as part of the full sustained measurement
    {
        const int N           = 1'000'000;
        const int WARMUP      = 1000;
        const int BURST_SIZE  = 10000;

        auto t_sustained_0 = std::chrono::steady_clock::now();

        for (int i = 0; i < WARMUP; ++i) {
            rlog::rangePush("bench", "external", "op", "");
            rlog::rangePop();
        }

        auto t_burst_0 = std::chrono::steady_clock::now();
        for (int i = 0; i < BURST_SIZE; ++i) {
            rlog::rangePush("bench", "external", "op", "");
            rlog::rangePop();
        }
        auto t_burst_1 = std::chrono::steady_clock::now();

        for (int i = WARMUP + BURST_SIZE; i < N; ++i) {
            rlog::rangePush("bench", "external", "op", "");
            rlog::rangePop();
        }

        auto t_sustained_1 = std::chrono::steady_clock::now();

        double burst_ms  = std::chrono::duration<double, std::milli>(t_burst_1 - t_burst_0).count();
        double sust_ms   = std::chrono::duration<double, std::milli>(t_sustained_1 - t_sustained_0).count();

        printf("burst:     %7d pairs  %8.1f ms  (%5.0f ns/pair  %.2f Mpairs/sec)"
               "  [after %d warmup]\n",
               BURST_SIZE, burst_ms, burst_ms * 1e6 / BURST_SIZE,
               BURST_SIZE / (burst_ms * 1e3), WARMUP);
        printf("sustained: %7d pairs  %8.1f ms  (%5.0f ns/pair  %.2f Mpairs/sec)\n",
               N, sust_ms, sust_ms * 1e6 / N, N / (sust_ms * 1e3));
    }

    return 0;
}
