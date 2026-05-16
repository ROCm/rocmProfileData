/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Utility.h"
#include "Firefly.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>

using namespace rpdtracer;
using namespace rpdtracer::firefly;

static void test_seq_zero_returns_zero()
{
    g_svcState.sequence.store(0, std::memory_order_relaxed);
    g_svcState.offset = 0;
    g_svcState.drift = 0.0;

    int64_t result = svc_read_offset_ns(1000000000ULL);
    assert(result == 0);
    fprintf(stderr, "  PASS: seq==0 fast path returns 0\n");
}

static void test_read_back_written_values()
{
    svc_update_ns(&g_svcState, 5000, 0.0);

    int64_t result = svc_read_offset_ns(1000000000ULL);
    assert(result == 5000);
    fprintf(stderr, "  PASS: offset-only read back\n");
}

static void test_drift_correction()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    timestamp_t ref_ns = timespec_to_ns(ts);

    svc_update_ns(&g_svcState, 1000, 0.001);

    timestamp_t now_ns = ref_ns + 1000000;
    int64_t result = svc_read_offset_ns(now_ns);

    // offset + drift * elapsed ≈ 1000 + 0.001 * 1000000 = 2000
    // Allow some tolerance for time between svc_update_ns and our ref_ns
    assert(result > 1500 && result < 2500);
    fprintf(stderr, "  PASS: drift correction (result=%lld)\n", static_cast<long long>(result));
}

static void test_concurrent_no_torn_reads()
{
    std::atomic<bool> done{false};
    std::atomic<int> errors{0};

    // Writer alternates between two distinct states with pauses
    // so readers can complete seqlock reads between writes
    std::thread writer([&]() {
        for (int i = 0; i < 10000 && !done.load(std::memory_order_relaxed); ++i) {
            if (i % 2 == 0)
                svc_update_ns(&g_svcState, 0, 0.0);
            else
                svc_update_ns(&g_svcState, 1000000, 0.0);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        done.store(true, std::memory_order_relaxed);
    });

    // Readers verify result is one of the two expected values
    auto reader_fn = [&]() {
        while (!done.load(std::memory_order_relaxed)) {
            timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            timestamp_t now = timespec_to_ns(ts);
            int64_t result = svc_read_offset_ns(now);
            // With drift=0, result should be exactly 0 or 1000000
            if (result != 0 && result != 1000000)
                errors.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i)
        readers.emplace_back(reader_fn);

    writer.join();
    for (auto& r : readers)
        r.join();

    assert(errors.load() == 0);
    fprintf(stderr, "  PASS: concurrent seqlock stress (4 readers, no torn reads)\n");
}

int main()
{
    fprintf(stderr, "test_seqlock:\n");
    test_seq_zero_returns_zero();
    test_read_back_written_values();
    test_drift_correction();
    test_concurrent_no_torn_reads();
    fprintf(stderr, "  All seqlock tests passed.\n");
    return 0;
}
