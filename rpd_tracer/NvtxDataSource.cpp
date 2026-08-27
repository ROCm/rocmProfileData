// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "NvtxDataSource.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <map>

#include <sqlite3.h>

#include "Logger.h"
#include "Utility.h"

using rpdtracer::DataSource;
using rpdtracer::NvtxDataSource;
using rpdtracer::NvtxDataSourcePrivate;
using rpdtracer::ApiTable;
using rpdtracer::Logger;
using rpdtracer::GetPid;
using rpdtracer::GetTid;
using rpdtracer::clocktime_ns;
using rpdtracer::timestamp_t;

namespace rpdtracer {

class NvtxDataSourcePrivate
{
public:
    std::atomic<bool> active{false};

    sqlite3_int64 domainId{0};
    sqlite3_int64 rangeCategoryId{0};
    sqlite3_int64 markCategoryId{0};
    sqlite3_int64 apiNameId{0};

    bool idsCached{false};
    void cacheIds();
};

}    // namespace rpdtracer


extern "C" {
    DataSource *NvtxDataSourceFactory() { return new NvtxDataSource(); }
}

static NvtxDataSource *s_instance = nullptr;

NvtxDataSource &NvtxDataSource::instance()
{
    return *s_instance;
}

NvtxDataSource::NvtxDataSource()
: d(new NvtxDataSourcePrivate())
{
}

NvtxDataSource::~NvtxDataSource()
{
    delete d;
}

// Per-thread nvtx range stack
static thread_local std::deque<ApiTable::row> t_nvtxStack;

// Track all thread stacks for shutdown drain
static std::mutex s_stacksMutex;
static std::map<std::pair<int,int>, std::deque<ApiTable::row>*> s_stacks;

static void registerThreadStack()
{
    static thread_local bool registered = false;
    if (!registered) {
        std::lock_guard<std::mutex> lock(s_stacksMutex);
        auto key = std::pair<int,int>(GetPid(), GetTid());
        s_stacks[key] = &t_nvtxStack;
        registered = true;
    }
}

static void drainStacks()
{
    timestamp_t now = clocktime_ns();
    Logger &logger = Logger::singleton();

    std::lock_guard<std::mutex> lock(s_stacksMutex);
    for (auto &entry : s_stacks) {
        auto &stack = *entry.second;
        while (!stack.empty()) {
            ApiTable::row row = stack.front();
            stack.pop_front();
            row.end = now;
            row.api_id = Logger::singleton().nextAnnotationId();
            logger.apiTable().insert(row);
        }
    }
}


void NvtxDataSourcePrivate::cacheIds()
{
    if (idsCached)
        return;
    Logger &logger = Logger::singleton();
    domainId = logger.stringTable().getOrCreate("nvtx");
    rangeCategoryId = logger.stringTable().getOrCreate("range");
    markCategoryId = logger.stringTable().getOrCreate("mark");
    apiNameId = logger.stringTable().getOrCreate("UserMarker");
    idsCached = true;
}

// ---- nvtx shim functions ----


extern "C" {

void nvtxMarkA(const char *message)
{
    NvtxDataSourcePrivate *d = NvtxDataSource::instance().priv();
    if (!d->active.load(std::memory_order_relaxed))
        return;
    d->cacheIds();

    Logger &logger = Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    row.domain_id = d->domainId;
    row.category_id = d->markCategoryId;
    row.apiName_id = d->apiNameId;
    row.args_id = logger.ustringTable().create(message);
    row.api_id = Logger::singleton().nextAnnotationId();

    logger.apiTable().insert(row);
}

int nvtxRangePushA(const char *message)
{
    NvtxDataSourcePrivate *d = NvtxDataSource::instance().priv();
    if (!d->active.load(std::memory_order_relaxed))
        return -1;
    d->cacheIds();

    registerThreadStack();

    Logger &logger = Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = 0;
    row.domain_id = d->domainId;
    row.category_id = d->rangeCategoryId;
    row.apiName_id = d->apiNameId;
    row.args_id = logger.ustringTable().create(message);
    row.api_id = 0;

    t_nvtxStack.push_front(row);
    return static_cast<int>(t_nvtxStack.size()) - 1;
}

int nvtxRangePop()
{
    NvtxDataSourcePrivate *d = NvtxDataSource::instance().priv();
    if (!d->active.load(std::memory_order_relaxed))
        return -1;

    if (t_nvtxStack.empty())
        return -1;

    Logger &logger = Logger::singleton();

    ApiTable::row row = t_nvtxStack.front();
    t_nvtxStack.pop_front();

    row.end = clocktime_ns();
    row.api_id = Logger::singleton().nextAnnotationId();

    logger.apiTable().insert(row);
    return static_cast<int>(t_nvtxStack.size());
}

}  // extern "C"


// ---- DataSource interface ----

void NvtxDataSource::init()
{
    s_instance = this;
}

void NvtxDataSource::startTracing()
{
    d->active.store(true, std::memory_order_release);
}

// Ranges do not span a stop/start.  stopTracing() closes every open range at
// the stop timestamp, so a push/pop pair straddling a pause is recorded as a
// range ending at the stop -- the pop after the restart finds an empty stack
// and is dropped.  This is the same behaviour as RlogDataSource.
void NvtxDataSource::stopTracing()
{
    d->active.store(false, std::memory_order_relaxed);

    // Drain in-flight ranges so their string ids (valid only in the
    // current storage) cannot outlive a resetStorage()
    drainStacks();
}

void NvtxDataSource::flush()
{
}

void NvtxDataSource::reset()
{
    d->idsCached = false;
}

void NvtxDataSource::end()
{
    d->active.store(false, std::memory_order_relaxed);
    drainStacks();
}
