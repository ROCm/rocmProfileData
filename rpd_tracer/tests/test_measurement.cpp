/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Firefly.h"

#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

using namespace rpdtracer::firefly;

static Measurement make_measurement(int64_t offset, char node)
{
    Measurement m{};
    m.offset = offset;
    m.node[0] = node;
    m.node[1] = '\0';
    return m;
}

static void test_basic_insert()
{
    MeasurementBuffer buf(100);
    assert(buf.count == 0);

    {
        std::lock_guard<std::mutex> lock(buf.mutex);
        buf.measurements[buf.count] = make_measurement(42, 'A');
        ++buf.count;
    }

    assert(buf.count == 1);
    assert(buf.measurements[0].offset == 42);
    assert(buf.measurements[0].node[0] == 'A');
    fprintf(stderr, "  PASS: basic insert\n");
}

static void test_concurrent_inserts()
{
    const int threads = 8;
    const int per_thread = 1000;
    MeasurementBuffer buf(threads * per_thread);

    auto worker = [&](int id) {
        for (int i = 0; i < per_thread; ++i) {
            Measurement m = make_measurement(id * per_thread + i, 'A');
            std::lock_guard<std::mutex> lock(buf.mutex);
            if (buf.count < static_cast<int>(buf.measurements.size())) {
                buf.measurements[buf.count] = m;
                ++buf.count;
            }
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i)
        workers.emplace_back(worker, i);
    for (auto& w : workers)
        w.join();

    assert(buf.count == threads * per_thread);
    fprintf(stderr, "  PASS: concurrent inserts (%d threads x %d = %d total)\n",
            threads, per_thread, buf.count);
}

static void test_capacity_limit()
{
    MeasurementBuffer buf(10);

    for (int i = 0; i < 20; ++i) {
        std::lock_guard<std::mutex> lock(buf.mutex);
        if (buf.count < static_cast<int>(buf.measurements.size())) {
            buf.measurements[buf.count] = make_measurement(i, 'B');
            ++buf.count;
        }
    }

    assert(buf.count == 10);
    fprintf(stderr, "  PASS: capacity limit (10 stored, 10 rejected)\n");
}

int main()
{
    fprintf(stderr, "test_measurement:\n");
    test_basic_insert();
    test_concurrent_inserts();
    test_capacity_limit();
    fprintf(stderr, "  All measurement tests passed.\n");
    return 0;
}
