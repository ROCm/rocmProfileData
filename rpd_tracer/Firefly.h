#ifndef FIREFLY_H
#define FIREFLY_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <linux/net_tstamp.h>
#include "Utility.h"

struct sockaddr_in;

namespace rpdtracer {
namespace firefly {

using timestamp_t = uint64_t;

// FIXME: use rlog properties in their own domain
// FIXME: replace batch regression with a PID controller for smooth, continuous
// clock correction. The PID naturally slews (no offset discontinuities) and
// eliminates the need for windowed regression entirely.
constexpr int CLOCKSYNC_PORT_DEFAULT = 29123;
constexpr std::size_t NETWORK_BUFFER_SIZE = 1024U;
constexpr int SOCKET_TIMEOUT_MSEC = 200;
constexpr int CONNECTION_RETRY_LIMIT = 30;
constexpr int CONNECTION_RETRY_DELAY_SEC = 1;
constexpr std::size_t MAX_MEASUREMENT_COUNT = 100000U;
constexpr std::size_t REGRESSION_WINDOW_SIZE = 200U;
constexpr double CONSENSUS_ALPHA = 0.5;
constexpr double GAIN_PHASE = 1.0;
constexpr double GAIN_FREQ = 1.0;
constexpr int FIRE_FLY_SLEEP_MSEC = 100;
constexpr int SYNC_TIMEOUT_SEC = 60;

struct Measurement {
    timestamp_t timestampNs;
    timestamp_t sendTimeA;
    timestamp_t recvTimeA;
    timestamp_t sendTimeB;
    timestamp_t recvTimeB;
    timestamp_t roundTripTime;
    int64_t     offset;
    int         udpPort;
    char        node[2];
};

struct MeasurementAnalysis {
    std::vector<Measurement> samples;
    int64_t averageOffset{0};
    double  driftRate{0.0};
};

struct MeasurementBuffer {
    std::vector<Measurement> measurements;
    int count{0};
    std::mutex mutex;

    MeasurementBuffer(std::size_t capacity) : measurements(capacity) {}
};

void run_probes(const char* role, const char* peerIp, int udpPort,
                MeasurementBuffer& buffer, std::atomic<bool>& done);

void run_server(int port, MeasurementBuffer& buffer, std::atomic<bool>& done);

void run_client(const char* serverIp, int port,
                MeasurementBuffer& buffer, std::atomic<bool>& done);

void firefly_run(const char* role, MeasurementBuffer& buffer);

timespec hw_now();

void svc_update_ns(SvcState* state, int64_t offsetNs, double drift);

MeasurementAnalysis read_latest_measurements(const char* role,
                                             std::size_t windowSize,
                                             const Measurement* measurements,
                                             int count,
                                             std::size_t capacity);
} // namespace firefly
} // namespace rpdtracer

#endif // FIREFLY_H
