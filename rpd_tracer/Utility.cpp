/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/

#include "Utility.h"
#include <atomic>

// create a namespace to hold the atomic offset, atomic drift and atomic timestamp variables
namespace chrono_sync {
    std::atomic<int64_t> offset(0);
    std::atomic<int64_t> drift(0);
    std::atomic<int64_t> last_host_timestamp(0);
}

void update_clock_offset(int64_t new_offset) {
    // Simple atomic store
    chrono_sync::offset.store(new_offset, std::memory_order_relaxed);
}

void adjust_offset_by_delta(int64_t delta) {
    // Atomic fetch and add
    chrono_sync::offset.fetch_add(delta, std::memory_order_relaxed);
}

void update_drift(int64_t new_drift) {
    chrono_sync::drift.store(new_drift, std::memory_order_relaxed);
}

int64_t get_current_offset() {
    // Atomic load
    return chrono_sync::offset.load(std::memory_order_relaxed);
}

// Example: Update all sync parameters atomically
void sync_clock(int64_t measured_offset, int64_t measured_drift) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    chrono_sync::offset.store(measured_offset, std::memory_order_release);
    chrono_sync::drift.store(measured_drift, std::memory_order_release);
    chrono_sync::last_host_timestamp.store(timespec_to_ns(ts), std::memory_order_release);
}
