/*********************************************************************************
* Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
********************************************************************************/
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
