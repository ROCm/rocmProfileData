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
#include "Utility.h"

#include <sqlite3.h>
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

// Known config properties (property name / env var):
//   filename       / RPDT_FILENAME       — output file path (default: ./trace.rpd)
//   delayinit      / RPDT_DELAYINIT      — skip singleton creation at load time (0/1, default: 0)
//   directwrite    / RPDT_DIRECTWRITE    — write directly to sqlite, no temp tables (0/1, default: 0)
//   autostart      / RPDT_AUTOSTART      — begin tracing immediately on init (0/1, default: 1)
//   autoflush      / RPDT_AUTOFLUSH      — periodic flush frequency in Hz (0=off, default: 0)
//   datasources_priority / RPDT_DATASOURCES_PRIORITY — comma-separated DataSource names to prioritize
//   datasources_explicit / RPDT_DATASOURCES_EXPLICIT — use only these DataSources (nothing else)
//   datasources_exclude  / RPDT_DATASOURCES_EXCLUDE  — remove these DataSources from the list
//   stackframes    / RPDT_STACKFRAMES    — record call stacks (0/1, default: 0)
//   rocprof_noargs / RPDT_ROCPROF_NOARGS — suppress rocprofiler kernel args (0/1, default: 0)
//   quiet          / RPDT_QUIET          — suppress informational output to stderr (0/1, default: 0)
//
// Embedded usage: set autostart=0 before the first rpdstart() call.
// autostart=1 (default) holds its own ref on the tracing state, so a
// subsequent rpdstop() will not actually stop tracing.
void rpd_setConfig(const char *property, const char *value)
{
    rpdtracer::setConfig(property, value);
}

sqlite3 *rpd_getConnection()
{
    return Logger::singleton().getConnection();
}

void rpd_resetStorage()
{
    Logger::singleton().resetStorage();
}

}  // extern "C"


#ifdef RPD_TRACER_BUILD

static void ourAtexitHandler(void*);

static void rpdTracerInit() __attribute__((constructor));
static void rpdTracerFinalize() __attribute__((destructor));

namespace {
    using real_cxa_atexit_t = int(*)(void(*)(void*), void*, void*);
    using real_atexit_t = int(*)(void(*)());
    real_cxa_atexit_t real_cxa_atexit = nullptr;
    real_atexit_t real_atexit_fn = nullptr;
    bool s_inInterceptor = false;
}

static void rpdTracerInit()
{
    Logger::rpdInit();

    real_cxa_atexit = (real_cxa_atexit_t)dlsym(RTLD_NEXT, "__cxa_atexit");
    real_atexit_fn = (real_atexit_t)dlsym(RTLD_NEXT, "atexit");
    real_cxa_atexit(ourAtexitHandler, nullptr, nullptr);
}

static void rpdTracerFinalize()
{
    Logger::rpdFinalize();
}

static void ourAtexitHandler(void*)
{
    Logger::rpdFinalize();
}

extern "C" int atexit(void (*fn)())
{
    if (real_atexit_fn == nullptr || s_inInterceptor)
        return 0;
    s_inInterceptor = true;
    real_atexit_fn(fn);
    real_cxa_atexit(ourAtexitHandler, nullptr, nullptr);
    s_inInterceptor = false;
    return 0;
}

extern "C" int __cxa_atexit(void (*func)(void*), void* arg, void* dso_handle)
{
    if (real_cxa_atexit == nullptr || s_inInterceptor)
        return 0;
    s_inInterceptor = true;
    real_cxa_atexit(func, arg, dso_handle);
    real_cxa_atexit(ourAtexitHandler, nullptr, nullptr);
    s_inInterceptor = false;
    return 0;
}

#else // embedded build

namespace {
    struct EmbeddedDefaults {
        EmbeddedDefaults() {
            rpdtracer::setConfig("autostart", "0");
        }
    } s_embeddedDefaults;
}

#endif // RPD_TRACER_BUILD
