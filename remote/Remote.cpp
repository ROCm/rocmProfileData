// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <dlfcn.h>
#include <signal.h>

static void remoteInit() __attribute__((constructor));

namespace {
    bool init = false;
    int refCount = 0;
    void (*dl) = nullptr;
};

void startTracing(int sig);
void stopTracing(int sig);

void remoteInit()
{
    fprintf(stderr, "rpdRemote: init()\n");
    signal(SIGUSR1, startTracing);
    signal(SIGUSR2, stopTracing);
}

void startTracing(int sig)
{
    signal(SIGUSR1, startTracing);
    if (refCount > 0)
        return;
    if (dl == nullptr) {
        dl = dlopen("librpd_tracer.so", RTLD_LAZY);
    }
    if (dl) {
        void (*start_func) (void) = reinterpret_cast<void(*)()>(dlsym(dl, "rpdstart"));
        if (start_func) {
            fprintf(stderr, "rpdRemote: tracing started\n");
            start_func();
        }
        else {
            fprintf(stderr, "rpdRemote: tracing failed\n");
        }
    }
    ++refCount;
}

void stopTracing(int sig)
{
    signal(SIGUSR2, stopTracing);
    if (refCount > 1)
        return;
    if (dl) {
        void (*stop_func) (void) = reinterpret_cast<void(*)()>(dlsym(dl, "rpdstop"));
        if (stop_func) {
            fprintf(stderr, "rpdRemote: tracing stopped\n");
            stop_func();
        }
        void (*flush_func) (void) = reinterpret_cast<void(*)()>(dlsym(dl, "rpdflush"));
        if (flush_func) {
            fprintf(stderr, "rpdRemote: trace flushed\n");
            flush_func();
        }
        // FIXME unloading is tricky, so don't
#if 0
        int ret = dlclose(dl);
        if (ret == 0)
            dl = nullptr;
#endif
        --refCount;
    }
}
