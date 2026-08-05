// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guard benchmark: measures the cost of guarding rangePush/rangePop with a
// cached bool (isLogging) when no tool is attached, so the calls are never
// made. A callback keeps isLogging in sync with the Hub's active state.
// Compare against bench_external to see how much the guard saves.

#include <rlog/client.h>

#include <atomic>
#include <chrono>
#include <cstdio>

static std::atomic<bool> isLogging{false};

static void onActiveChanged()
{
    isLogging = rlog::isActive();
}

int main()
{
    rlog::init();
    rlog::registerActiveCallback(onActiveChanged);

    printf("guard: isLogging = %s\n\n", isLogging ? "true" : "false (baseline)");

    // --- sustained + burst window ---
    {
        const int N          = 1'000'000;
        const int WARMUP     = 1000;
        const int BURST_SIZE = 10000;

        auto t_sustained_0 = std::chrono::steady_clock::now();

        for (int i = 0; i < WARMUP; ++i) {
            if (isLogging) {
                rlog::rangePush("bench", "guard", "op", "");
                rlog::rangePop();
            }
        }

        auto t_burst_0 = std::chrono::steady_clock::now();
        for (int i = 0; i < BURST_SIZE; ++i) {
            if (isLogging) {
                rlog::rangePush("bench", "guard", "op", "");
                rlog::rangePop();
            }
        }
        auto t_burst_1 = std::chrono::steady_clock::now();

        for (int i = WARMUP + BURST_SIZE; i < N; ++i) {
            if (isLogging) {
                rlog::rangePush("bench", "guard", "op", "");
                rlog::rangePop();
            }
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
