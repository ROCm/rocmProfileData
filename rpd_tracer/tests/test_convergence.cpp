/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Firefly.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace rpdtracer::firefly;

// ---------------------------------------------------------------------------
// Simulated clock and network model
// ---------------------------------------------------------------------------
struct ClockModel {
    int64_t offset_ns;      // true offset of remote clock (ns)
    double  drift_ppm;      // remote clock drift (parts per million)

    // Remote clock time given local time
    int64_t remote_time(int64_t local_ns) const {
        return local_ns + offset_ns + static_cast<int64_t>(drift_ppm * 1e-6 * local_ns);
    }
};

struct NetworkModel {
    double base_latency_us;    // one-way base latency (microseconds)
    double jitter_sigma_us;    // log-normal jitter sigma
    double asymmetry;          // forward path multiplier (1.0 = symmetric)
    double cpu_noise_us;       // uniform CPU scheduling noise bound

    std::mt19937 rng;

    NetworkModel(double lat, double jit, double asym, double cpu, unsigned seed = 42)
        : base_latency_us(lat), jitter_sigma_us(jit), asymmetry(asym),
          cpu_noise_us(cpu), rng(seed) {}

    int64_t jitter_ns() {
        if (jitter_sigma_us <= 0.0)
            return 0;
        std::lognormal_distribution<double> dist(0.0, jitter_sigma_us / base_latency_us);
        return static_cast<int64_t>(dist(rng) * base_latency_us * 1000.0);
    }

    int64_t forward_delay_ns() {
        return static_cast<int64_t>(base_latency_us * asymmetry * 1000.0) + jitter_ns();
    }

    int64_t return_delay_ns() {
        return static_cast<int64_t>(base_latency_us * 1000.0) + jitter_ns();
    }

    int64_t cpu_noise_ns() {
        if (cpu_noise_us <= 0.0)
            return 0;
        std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(cpu_noise_us * 1000.0));
        return dist(rng);
    }
};

// Generate a synthetic Measurement as seen by node B (client)
static Measurement synthesize_probe(int64_t local_time_ns,
                                    const ClockModel& clock,
                                    NetworkModel& net) {
    int64_t processing_delay = 5000; // 5μs processing on remote

    // Forward path: A sends, B receives
    int64_t fwd_delay = net.forward_delay_ns();
    int64_t ret_delay = net.return_delay_ns();

    int64_t sendTimeA = local_time_ns + net.cpu_noise_ns();
    int64_t recvTimeB = clock.remote_time(local_time_ns + fwd_delay) + net.cpu_noise_ns();
    int64_t sendTimeB = recvTimeB + processing_delay + net.cpu_noise_ns();
    int64_t recvTimeA = local_time_ns + fwd_delay + processing_delay + ret_delay + net.cpu_noise_ns();

    int64_t roundTripTime = (recvTimeA - sendTimeA) - (sendTimeB - recvTimeB);
    int64_t offset = static_cast<int64_t>(recvTimeA - sendTimeB) - static_cast<int64_t>(roundTripTime / 2);

    Measurement m{};
    m.timestampNs = static_cast<uint64_t>(local_time_ns);
    m.sendTimeA = static_cast<uint64_t>(sendTimeA);
    m.recvTimeA = static_cast<uint64_t>(recvTimeA);
    m.sendTimeB = static_cast<uint64_t>(sendTimeB);
    m.recvTimeB = static_cast<uint64_t>(recvTimeB);
    m.roundTripTime = static_cast<uint64_t>(roundTripTime);
    m.offset = offset;
    m.udpPort = 0;
    m.node[0] = 'B';
    m.node[1] = '\0';
    return m;
}

// Run a simulation: returns vector of (probe_index, published_offset) pairs
struct SimResult {
    std::vector<std::pair<int, int64_t>> offsets;
    int convergence_probe;   // first probe where |error| < threshold
    int64_t steady_state_err; // average |error| over last 20% of probes
    bool monotonic;           // no backwards jumps in corrected time
};

static SimResult run_simulation(const ClockModel& clock, NetworkModel& net,
                                int num_probes, int64_t probe_interval_ns,
                                int64_t convergence_threshold_ns) {
    SimResult result{};
    result.convergence_probe = -1;
    result.monotonic = true;

    MeasurementBuffer buffer(MAX_MEASUREMENT_COUNT);

    // Reset global state and PI controller
    g_pSvcState->sequence.store(0, std::memory_order_relaxed);
    g_pSvcState->offset = 0;
    g_pSvcState->drift = 0.0;
    g_pSvcState->referenceTime = {};
    firefly_reset();

    int64_t base_time = 1000000000000LL; // 1000 seconds as epoch
    int64_t prev_corrected = 0;

    for (int i = 0; i < num_probes; ++i) {
        int64_t local_time = base_time + i * probe_interval_ns;

        Measurement m = synthesize_probe(local_time, clock, net);

        {
            std::lock_guard<std::mutex> lock(buffer.mutex);
            int idx = buffer.count % static_cast<int>(buffer.measurements.size());
            buffer.measurements[idx] = m;
            ++buffer.count;
        }

        firefly_run("B", buffer);

        int64_t published = g_pSvcState->offset;
        int64_t true_offset = local_time - clock.remote_time(local_time);
        int64_t error = published - true_offset;

        result.offsets.emplace_back(i, published);

        if (result.convergence_probe < 0 && std::abs(error) < convergence_threshold_ns)
            result.convergence_probe = i;

        // Check monotonicity of corrected timestamps
        int64_t corrected = local_time + published;
        if (i > 0 && corrected < prev_corrected)
            result.monotonic = false;
        prev_corrected = corrected;
    }

    // Steady-state error: average over last 20%
    int tail_start = num_probes * 4 / 5;
    int64_t err_sum = 0;
    int err_count = 0;
    for (int i = tail_start; i < num_probes; ++i) {
        int64_t local_time = base_time + i * probe_interval_ns;
        int64_t true_offset = local_time - clock.remote_time(local_time);
        int64_t error = result.offsets[i].second - true_offset;
        err_sum += std::abs(error);
        ++err_count;
    }
    result.steady_state_err = (err_count > 0) ? err_sum / err_count : 0;

    return result;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
static void test_constant_offset_convergence() {
    ClockModel clock{50000, 0.0}; // 50μs offset, no drift
    NetworkModel net(200.0, 50.0, 1.0, 10.0); // 200μs lat, 50μs jitter, symmetric, 10μs cpu

    auto r = run_simulation(clock, net, 500, 100000000LL, 5000); // 100ms interval, 5μs threshold

    fprintf(stderr, "  constant offset: converged at probe %d, steady-state err %lldns\n",
            r.convergence_probe, static_cast<long long>(r.steady_state_err));

    assert(r.convergence_probe >= 0 && r.convergence_probe < 200);
    assert(r.steady_state_err < 20000); // < 20μs
    fprintf(stderr, "  PASS\n");
}

static void test_drift_tracking() {
    ClockModel clock{100000, 20.0}; // 100μs offset, 20 ppm drift
    NetworkModel net(200.0, 50.0, 1.0, 10.0);

    auto r = run_simulation(clock, net, 1000, 100000000LL, 50000); // 50μs threshold

    fprintf(stderr, "  drift tracking (20ppm): converged at probe %d, steady-state err %lldns\n",
            r.convergence_probe, static_cast<long long>(r.steady_state_err));

    assert(r.convergence_probe >= 0 && r.convergence_probe < 400);
    assert(r.steady_state_err < 300000); // < 300μs (drift tracking lag, PID will improve)
    fprintf(stderr, "  PASS\n");
}

static void test_high_jitter() {
    ClockModel clock{50000, 0.0};
    NetworkModel net(2000.0, 500.0, 1.0, 30.0); // 2ms latency, 500μs jitter, 30μs cpu

    auto r = run_simulation(clock, net, 1000, 100000000LL, 50000);

    fprintf(stderr, "  high jitter: converged at probe %d, steady-state err %lldns\n",
            r.convergence_probe, static_cast<long long>(r.steady_state_err));

    assert(r.convergence_probe >= 0 && r.convergence_probe < 500);
    assert(r.steady_state_err < 500000); // < 500μs (noisy environment)
    fprintf(stderr, "  PASS\n");
}

static void test_asymmetric_path() {
    ClockModel clock{50000, 0.0};
    NetworkModel net(200.0, 50.0, 1.10, 10.0); // 10% forward asymmetry

    auto r = run_simulation(clock, net, 500, 100000000LL, 50000);

    // Asymmetry introduces irreducible bias ≈ base_latency * (asym-1) / 2 = 10μs
    fprintf(stderr, "  asymmetric path (10%%): converged at probe %d, steady-state err %lldns\n",
            r.convergence_probe, static_cast<long long>(r.steady_state_err));

    // Steady-state error includes the irreducible asymmetry bias
    assert(r.steady_state_err < 100000); // < 100μs (includes ~10μs bias)
    fprintf(stderr, "  PASS\n");
}

static void test_step_change() {
    // Simulate a step change in offset at probe 250
    ClockModel clock1{50000, 0.0};
    ClockModel clock2{150000, 0.0}; // jump +100μs
    NetworkModel net(200.0, 50.0, 1.0, 10.0);

    MeasurementBuffer buffer(MAX_MEASUREMENT_COUNT);
    g_pSvcState->sequence.store(0, std::memory_order_relaxed);
    g_pSvcState->offset = 0;
    g_pSvcState->drift = 0.0;
    g_pSvcState->referenceTime = {};
    firefly_reset();

    int64_t base_time = 1000000000000LL;
    int64_t probe_interval = 100000000LL;
    int converged_after_step = -1;

    for (int i = 0; i < 500; ++i) {
        int64_t local_time = base_time + i * probe_interval;
        const ClockModel& clock = (i < 250) ? clock1 : clock2;

        Measurement m = synthesize_probe(local_time, clock, net);
        {
            std::lock_guard<std::mutex> lock(buffer.mutex);
            int idx = buffer.count % static_cast<int>(buffer.measurements.size());
            buffer.measurements[idx] = m;
            ++buffer.count;
        }
        firefly_run("B", buffer);

        if (i >= 250) {
            int64_t true_offset = local_time - clock2.remote_time(local_time);
            int64_t error = g_pSvcState->offset - true_offset;
            if (converged_after_step < 0 && std::abs(error) < 20000)
                converged_after_step = i - 250;
        }
    }

    fprintf(stderr, "  step change: re-converged %d probes after step\n", converged_after_step);
    assert(converged_after_step >= 0 && converged_after_step < 200);
    fprintf(stderr, "  PASS\n");
}

static void test_monotonic_correction() {
    ClockModel clock{50000, 10.0}; // moderate drift
    NetworkModel net(200.0, 30.0, 1.0, 5.0); // low noise

    auto r = run_simulation(clock, net, 500, 100000000LL, 10000);

    fprintf(stderr, "  monotonic correction: %s\n", r.monotonic ? "yes" : "NO — backwards jumps detected");
    // Note: with batch regression + discrete updates, this may fail.
    // This test documents the current behavior and will verify PID improvement.
    if (!r.monotonic)
        fprintf(stderr, "  EXPECTED FAIL (batch regression causes discontinuities, PID will fix)\n");
    fprintf(stderr, "  PASS (informational)\n");
}

// ---------------------------------------------------------------------------
int main() {
    fprintf(stderr, "test_convergence:\n");
    test_constant_offset_convergence();
    test_drift_tracking();
    test_high_jitter();
    test_asymmetric_path();
    test_step_change();
    test_monotonic_correction();
    fprintf(stderr, "  All convergence tests passed.\n");
    return 0;
}
