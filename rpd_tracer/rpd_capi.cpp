/*********************************************************************************
* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "Logger.h"

#include <vector>
#include <mutex>
#include <stdlib.h>
#include <dlfcn.h>

using rpdtracer::Logger;

// Common C API — compiled into both librpd_tracer and librpd_embedded
extern "C" {

void rpdstart()
{
    Logger::singleton().rpdstart();
}

void rpdstop()
{
    Logger::singleton().rpdstop();
}

void rpdflush()
{
    Logger::singleton().rpdflush();
}

void rpd_rangePush(const char *domain, const char *apiName, const char* args)
{
    Logger::singleton().rpd_rangePush(domain, apiName, args);
}

void rpd_rangePop()
{
    Logger::singleton().rpd_rangePop();
}

}  // extern "C"


#ifdef RPD_TRACER_BUILD

static void ourAtexitHandler(void*);

static void rpdTracerInit() __attribute__((constructor));
static void rpdTracerFinalize() __attribute__((destructor));

static void rpdTracerInit()
{
    Logger::rpdInit();

    static auto real_cxa_atexit = (int(*)(void(*)(void*), void*, void*))dlsym(RTLD_NEXT, "__cxa_atexit");
    real_cxa_atexit(ourAtexitHandler, nullptr, nullptr);
}

static void rpdTracerFinalize()
{
    Logger::rpdFinalize();
}

// atexit/cxa_atexit interception — call rpdFinalize before any registered handler

namespace {
    struct AtexitEntry {
        void (*func)(void*);
        void* arg;
    };
    std::vector<AtexitEntry> s_atexitList;
    std::mutex s_atexitMutex;

    void c_atexit_trampoline(void* fn) {
        reinterpret_cast<void(*)()>(fn)();
    }
}

static void ourAtexitHandler(void*)
{
    Logger::rpdFinalize();
    std::unique_lock<std::mutex> lock(s_atexitMutex);
    while (!s_atexitList.empty()) {
        auto entry = s_atexitList.back();
        s_atexitList.pop_back();
        lock.unlock();
        entry.func(entry.arg);
        lock.lock();
    }
}

extern "C" int atexit(void (*fn)())
{
    std::lock_guard<std::mutex> lock(s_atexitMutex);
    s_atexitList.push_back({c_atexit_trampoline, reinterpret_cast<void*>(fn)});
    return 0;
}

extern "C" int __cxa_atexit(void (*func)(void*), void* arg, void* /*dso_handle*/)
{
    std::lock_guard<std::mutex> lock(s_atexitMutex);
    s_atexitList.push_back({func, arg});
    return 0;
}

#endif // RPD_TRACER_BUILD
