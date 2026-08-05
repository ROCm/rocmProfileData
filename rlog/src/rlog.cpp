// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rlog/client.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <string>


using rlog::Api;

namespace {
    std::string domain;
    std::string category;

    // Static function pointers
    void (*log_mark_) (const char*, const char*, const char*, const char*) = NULL;
    void (*log_rangePush_) (const char*, const char*, const char*, const char*) = NULL;
    void (*log_rangePop_) () = NULL;
    void (*log_registerActiveCallback_) (void (*cb)()) = NULL;
    bool (*log_isActive_) () = NULL;
    const char* (*log_getProperty_) (const char*, const char*, const char*) = NULL;

    // Static function pointers - ROCTX
    void (*roctx_mark_) (const char* message) = NULL;
    void (*roctx_rangePush_) (const char* message) = NULL;
    void (*roctx_rangePop_) () = NULL;
    // Static function pointers - NVTX
    void (*nvtx_mark_) (const char* message) = NULL;
    void (*nvtx_rangePush_) (const char* message) = NULL;
    void (*nvtx_rangePop_) () = NULL;

    // Logging library state and location
    bool enabled[Api::API_COUNT] {};
    bool loaded[Api::API_COUNT] {};
    std::atomic<bool> active[Api::API_COUNT] {};
    std::string libnames[Api::API_COUNT] { "" };

    // Used with ENV vars to force legacy logging
    bool forceLog { false };

    std::mutex configMutex;

    // Load symbols for enabled libs
    void configure() {
        if (!loaded[Api::Rlog] && enabled[Api::Rlog]) {
            // load and lookup Rlog
            void (*dl) = dlopen(libnames[Api::Rlog].c_str(), RTLD_LAZY);
            if (dl) {
                log_mark_ = (void (*)(const char*, const char*, const char*, const char*)) dlsym(dl, "rlog_mark");
                log_rangePush_ = (void (*)(const char*, const char*, const char*, const char*)) dlsym(dl, "rlog_rangePush");
                log_rangePop_ = (void (*)()) dlsym(dl, "rlog_rangePop");
                log_registerActiveCallback_ = (void(*)(void (*cb)())) dlsym(dl, "rlog_registerActiveCallback");
                log_isActive_ = (bool (*)()) dlsym(dl, "rlog_isActive");
                log_getProperty_ = (const char*(*)(const char*, const char*, const char*))  dlsym(dl, "rlog_getProperty");
                if (log_mark_
                    && log_rangePush_
                    && log_rangePop_
                    && log_registerActiveCallback_
                    && log_isActive_
                    && log_getProperty_) {
                    loaded[Api::Rlog] = true;
                    active[Api::Rlog] = loaded[Api::Rlog] && ::enabled[Api::Rlog];
                }
            }
        }
        if (!loaded[Api::Roctx] && enabled[Api::Roctx]) {
            void (*dltx) = dlopen(libnames[Api::Roctx].c_str(), RTLD_LAZY);
            if (dltx) {
                roctx_mark_ = (void (*)(const char*)) dlsym(dltx, "roctxMarkA");
                roctx_rangePush_ = (void (*)(const char*)) dlsym(dltx, "roctxRangePushA");
                roctx_rangePop_ = (void (*)()) dlsym(dltx, "roctxRangePop");
                if (roctx_mark_
                    && roctx_rangePush_
                    && roctx_rangePop_) {
                    loaded[Api::Roctx] = true;
                    active[Api::Roctx] = loaded[Api::Roctx] && ::enabled[Api::Roctx];
                }
            }
        }
        if (!loaded[Api::Nvtx] && enabled[Api::Nvtx]) {
            void (*dltx) = dlopen(libnames[Api::Nvtx].c_str(), RTLD_LAZY);
            if (dltx) {
                nvtx_mark_ = (void (*)(const char*)) dlsym(dltx, "nvtxMarkA");
                nvtx_rangePush_ = (void (*)(const char*)) dlsym(dltx, "nvtxRangePushA");
                nvtx_rangePop_ = (void (*)()) dlsym(dltx, "nvtxRangePop");
                if (nvtx_mark_
                    && nvtx_rangePush_
                    && nvtx_rangePop_) {
                    loaded[Api::Nvtx] = true;
                    active[Api::Nvtx] = loaded[Api::Nvtx] && ::enabled[Api::Nvtx];
                }
            }
        }
    }
} // namespace

namespace rlog {

// Load library and look up symbols
void init() {
    // Load librlog.so first so log_getProperty_ is available for config lookup
    libnames[Api::Rlog] = "librlog.so";
    setEnabled(Api::Rlog, true);

    // Query each config property — INSERT OR IGNORE ensures defaults land in
    // the store on first run. Env var overrides the stored value at runtime.
    const char *roctxlib_prop    = log_getProperty_ ? log_getProperty_("rlog", "RLOG_ROCTX_LIBPATH", "librocprofiler-sdk-roctx.so") : "librocprofiler-sdk-roctx.so";
    const char *nvtxlib_prop     = log_getProperty_ ? log_getProperty_("rlog", "RLOG_NVTX_LIBPATH",  "libcupti.so")                 : "libcupti.so";
    const char *roctx_force_prop = log_getProperty_ ? log_getProperty_("rlog", "RLOG_FORCE_ROCTX",   "0")                           : "0";
    const char *nvtx_force_prop  = log_getProperty_ ? log_getProperty_("rlog", "RLOG_FORCE_NVTX",    "0")                           : "0";

    // Apply: env var takes priority over stored property
    const char *roctxlib = getenv("RLOG_ROCTX_LIBPATH");
    libnames[Api::Roctx] = roctxlib ? roctxlib : roctxlib_prop;

    const char *nvtxlib = getenv("RLOG_NVTX_LIBPATH");
    libnames[Api::Nvtx] = nvtxlib ? nvtxlib : nvtxlib_prop;

    const char *roctx_force = getenv("RLOG_FORCE_ROCTX");
    if (atoi(roctx_force ? roctx_force : roctx_force_prop) != 0) {
        setEnabled(Api::Roctx, true);
        forceLog = true;
    }

    const char *nvtx_force = getenv("RLOG_FORCE_NVTX");
    if (atoi(nvtx_force ? nvtx_force : nvtx_force_prop) != 0) {
        setEnabled(Api::Nvtx, true);
        forceLog = true;
    }
}

bool enabled(rlog::Api api)
{
    std::unique_lock<std::mutex> lock(configMutex);
    return ::enabled[api];
}

void setEnabled(rlog::Api api, bool enable)
{
    std::unique_lock<std::mutex> lock(configMutex);
    bool prev = ::enabled[api];
    ::enabled[api] = enable;
    if (prev == enable)
        return;
    if (enable)
        configure();  // reconfigure to load symbols - will become active if successful
    active[api] = loaded[api] && ::enabled[api];
}

void mark(const char *domain, const char *category, const char *apiname, const char *args)
{
    if (active[Api::Rlog])
        log_mark_(domain, category, apiname, args);
    if (active[Api::Roctx] || active[Api::Nvtx]) {
        char buff[4096];
        snprintf(buff, 4096, "%s : %s : api = %s | %s", domain, category, apiname, args);
        if (active[Api::Roctx])
            roctx_mark_(buff);
        if (active[Api::Nvtx]) 
            nvtx_mark_(buff);
    }
}

void mark(const char *category, const char *apiname, const char *args)
{
    mark(domain.c_str(), category, apiname, args);
}

void mark(const char *apiname, const char *args)
{
    mark(domain.c_str(), category.c_str(), apiname, args);
}

void rangePush(const char *domain, const char *category, const char *apiname, const char *args)
{
    if (active[Api::Rlog])
       log_rangePush_(domain, category, apiname, args);
    if (active[Api::Roctx] || active[Api::Nvtx]) {
        char buff[4096];
        snprintf(buff, 4096, "%s : %s : api = %s | %s", domain, category, apiname, args);
        if (active[Api::Roctx])
            roctx_rangePush_(buff);
        if (active[Api::Nvtx])
            nvtx_rangePush_(buff);
    }
}

void rangePush(const char *category, const char *apiname, const char *args)
{
    rangePush(domain.c_str(), category, apiname, args);
}

void rangePush(const char *apiname, const char *args)
{
    rangePush(domain.c_str(), category.c_str(), apiname, args);
}

void rangePop()
{
    if (active[Api::Rlog])
        log_rangePop_();
    if (active[Api::Roctx])
        roctx_rangePop_();
    if (active[Api::Nvtx])
        nvtx_rangePop_();
}

int registerActiveCallback(void (*cb)())
{
    if (log_registerActiveCallback_) {
        log_registerActiveCallback_(cb);
        return 0;
    }
    else
        return -1;
}

bool isActive()
{
    if (forceLog)
        return true;
    if (log_isActive_)
        return log_isActive_();
    else
        return false;
}

void setDefaultDomain(const char* ddomain)
{
    domain = ddomain ? ddomain : "";
}

void setDefaultCategory(const char* dcat)
{
    category = dcat ? dcat : "";
}

const char *getProperty(const char *domain, const char *property, const char *defaultValue)
{
    if (log_getProperty_)
        return log_getProperty_(domain, property, defaultValue);
    return defaultValue;
}

} // namespace rlog
