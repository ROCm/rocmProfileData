/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Firefly.h"

#include <cassert>
#include <cstdio>
#include <cmath>

using namespace rpdtracer::firefly;

static Measurement make_timed(timestamp_t ts, int64_t offset, char node)
{
    Measurement m{};
    m.timestampNs = ts;
    m.offset = offset;
    m.node[0] = node;
    m.node[1] = '\0';
    return m;
}

static void test_constant_offset()
{
    const int N = 100;
    Measurement data[N];
    for (int i = 0; i < N; ++i)
        data[i] = make_timed(1000000000ULL + i * 1000000ULL, 5000, 'A');

    MeasurementAnalysis a = read_latest_measurements("A", 1000, data, N);
    assert(a.samples.size() == static_cast<size_t>(N));
    assert(a.averageOffset == 5000);
    assert(std::fabs(a.driftRate) < 1e-12);
    fprintf(stderr, "  PASS: constant offset (avg=%lld, drift=%.2e)\n",
            static_cast<long long>(a.averageOffset), a.driftRate);
}

static void test_known_drift()
{
    const int N = 100;
    const double drift_per_ns = 0.0001;
    Measurement data[N];

    timestamp_t t0 = 1000000000ULL;
    for (int i = 0; i < N; ++i) {
        timestamp_t ts = t0 + i * 1000000ULL;
        int64_t offset = static_cast<int64_t>(1000 + drift_per_ns * (ts - t0));
        data[i] = make_timed(ts, offset, 'A');
    }

    MeasurementAnalysis a = read_latest_measurements("A", 1000, data, N);
    assert(a.samples.size() == static_cast<size_t>(N));
    double drift_error = std::fabs(a.driftRate - drift_per_ns);
    assert(drift_error < 1e-8);
    fprintf(stderr, "  PASS: known drift (expected=%.4e, got=%.4e, err=%.2e)\n",
            drift_per_ns, a.driftRate, drift_error);
}

static void test_role_filter()
{
    Measurement data[4];
    data[0] = make_timed(1000000000ULL, 100, 'A');
    data[1] = make_timed(1000000001ULL, 200, 'B');
    data[2] = make_timed(1000000002ULL, 300, 'A');
    data[3] = make_timed(1000000003ULL, 400, 'B');

    MeasurementAnalysis a = read_latest_measurements("A", 1000, data, 4);
    assert(a.samples.size() == 2);
    assert(a.samples[0].offset == 100);
    assert(a.samples[1].offset == 300);

    MeasurementAnalysis b = read_latest_measurements("B", 1000, data, 4);
    assert(b.samples.size() == 2);
    assert(b.samples[0].offset == 200);
    assert(b.samples[1].offset == 400);

    fprintf(stderr, "  PASS: role filter\n");
}

static void test_empty_input()
{
    MeasurementAnalysis a = read_latest_measurements("A", 1000, nullptr, 0);
    assert(a.samples.empty());
    assert(a.averageOffset == 0);
    assert(a.driftRate == 0.0);
    fprintf(stderr, "  PASS: empty input\n");
}

static void test_single_measurement()
{
    Measurement data[1];
    data[0] = make_timed(1000000000ULL, 42, 'A');

    MeasurementAnalysis a = read_latest_measurements("A", 1000, data, 1);
    assert(a.samples.size() == 1);
    assert(a.averageOffset == 42);
    assert(a.driftRate == 0.0);
    fprintf(stderr, "  PASS: single measurement\n");
}

static void test_window_truncation()
{
    const int N = 100;
    Measurement data[N];
    for (int i = 0; i < N; ++i)
        data[i] = make_timed(1000000000ULL + i * 1000000ULL, i, 'A');

    MeasurementAnalysis a = read_latest_measurements("A", 10, data, N);
    assert(a.samples.size() == 10);
    assert(a.samples[0].offset == 90);
    assert(a.samples[9].offset == 99);
    fprintf(stderr, "  PASS: window truncation (100 -> 10)\n");
}

static void test_drift_clamp()
{
    // Build a buffer with measurements that produce drift > 500 ppm
    const int N = 100;
    const double bad_drift = 0.01;  // 10,000,000 ppm — way over 500 ppm
    MeasurementBuffer buf(N);

    timestamp_t t0 = 1000000000ULL;
    for (int i = 0; i < N; ++i) {
        timestamp_t ts = t0 + i * 1000000ULL;
        int64_t offset = static_cast<int64_t>(bad_drift * (ts - t0));
        buf.measurements[i] = make_timed(ts, offset, 'A');
    }
    buf.count = N;

    // Reset g_svcState
    g_pSvcState->sequence.store(0, std::memory_order_relaxed);
    g_pSvcState->offset = 0;
    g_pSvcState->drift = 0.0;

    firefly_run("A", buf);

    // Drift should have been clamped to 0
    assert(std::fabs(g_pSvcState->drift) < 1e-15);
    fprintf(stderr, "  PASS: drift clamp (bad drift=%.2e clamped to 0)\n", bad_drift);
}

int main()
{
    fprintf(stderr, "test_regression:\n");
    test_constant_offset();
    test_known_drift();
    test_role_filter();
    test_empty_input();
    test_single_measurement();
    test_window_truncation();
    test_drift_clamp();
    fprintf(stderr, "  All regression tests passed.\n");
    return 0;
}
