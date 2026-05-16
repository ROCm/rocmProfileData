#ifndef FIREFLY_H
#define FIREFLY_H

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
constexpr int TCP_PORT_DEFAULT = 12345;
constexpr int UDP_PORT_DEFAULT = 12345;
constexpr std::size_t NETWORK_BUFFER_SIZE = 1024U;
constexpr int SOCKET_TIMEOUT_MSEC = 200;
constexpr int CONNECTION_RETRY_LIMIT = 30;
constexpr int CONNECTION_RETRY_DELAY_SEC = 1;
constexpr std::size_t MAX_MEASUREMENT_COUNT = 100000U;
constexpr std::size_t REGRESSION_WINDOW_SIZE = 1000U;
constexpr double CONSENSUS_ALPHA = 0.5;
constexpr double GAIN_PHASE = 1.0;
constexpr double GAIN_FREQ = 1.0;
constexpr int FIRE_FLY_SLEEP_MSEC = 100;

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

bool enable_sw_timestamps(int sockFd);

bool send_probe(int sockFd, sockaddr_in* peerAddress, timestamp_t* sendTime, int probeId);

void receive_probe(int sockFd, timestamp_t* receiveTime, int expectedProbeId);

int tcp_handshake(const char* role, const char* peerIp, int tcpPort);

void run_node(const char* role,
              const char* ipA,
              const char* ipB,
              int udpPort,
              MeasurementBuffer& buffer,
              bool& done);

void firefly_run(const char* role, MeasurementBuffer& buffer);

timespec hw_now();

void svc_update_ns(SvcState* state, int64_t offsetNs, double drift);

MeasurementAnalysis read_latest_measurements(const char* role,
                                             std::size_t windowSize,
                                             const Measurement* measurements,
                                             int count);
} // namespace firefly
} // namespace rpdtracer

#endif // FIREFLY_H
