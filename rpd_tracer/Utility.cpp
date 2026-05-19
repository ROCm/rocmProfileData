/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/

#include "Utility.h"

#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace rpdtracer {

int sqlite_busy_handler(void *data, int count)
{
    count = (count < 9) ? count : 8;
    usleep(1000 * (0x1 << count));
    return 1;
}

namespace firefly {

static SvcState s_localState;
SvcState* g_pSvcState = &s_localState;

static void* s_shmAddr = nullptr;

void create_clocksync_shm(std::string& shm_name_out) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    char name[64];
    std::snprintf(name, sizeof(name), "/rpd_sync_%d_%lld",
                  static_cast<int>(getpid()),
                  static_cast<long long>(ts.tv_sec * 1000000000LL + ts.tv_nsec));
    shm_name_out = name;

    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd < 0) {
        std::fprintf(stderr, "ChronoSync: shm_open create failed (errno: %s)\n", std::strerror(errno));
        return;
    }

    if (ftruncate(fd, sizeof(SvcState)) < 0) {
        std::fprintf(stderr, "ChronoSync: ftruncate failed (errno: %s)\n", std::strerror(errno));
        close(fd);
        shm_unlink(name);
        return;
    }

    void* addr = mmap(nullptr, sizeof(SvcState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::fprintf(stderr, "ChronoSync: mmap failed (errno: %s)\n", std::strerror(errno));
        shm_unlink(name);
        return;
    }

    std::memset(addr, 0, sizeof(SvcState));
    new (addr) SvcState();

    s_shmAddr = addr;
    g_pSvcState = static_cast<SvcState*>(addr);
    std::fprintf(stderr, "ChronoSync: created shm %s\n", name);
}

void attach_clocksync_shm(const std::string& shm_name) {
    int fd = shm_open(shm_name.c_str(), O_RDWR, 0);
    if (fd < 0) {
        std::fprintf(stderr, "ChronoSync: shm_open attach failed for %s (errno: %s)\n",
                     shm_name.c_str(), std::strerror(errno));
        return;
    }

    void* addr = mmap(nullptr, sizeof(SvcState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::fprintf(stderr, "ChronoSync: mmap attach failed (errno: %s)\n", std::strerror(errno));
        return;
    }

    s_shmAddr = addr;
    g_pSvcState = static_cast<SvcState*>(addr);
    std::fprintf(stderr, "ChronoSync: attached shm %s\n", shm_name.c_str());
}

void cleanup_clocksync_shm(const std::string& shm_name) {
    if (s_shmAddr != nullptr) {
        munmap(s_shmAddr, sizeof(SvcState));
        s_shmAddr = nullptr;
    }
    g_pSvcState = &s_localState;

    if (!shm_name.empty())
        shm_unlink(shm_name.c_str());

    std::fprintf(stderr, "ChronoSync: cleaned up shm %s\n", shm_name.c_str());
}

} // namespace firefly
} // namespace rpdtracer
