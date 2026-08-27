// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "RoctxDataSource.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <map>

#include <sqlite3.h>

#include "Logger.h"
#include "Utility.h"

using rpdtracer::DataSource;
using rpdtracer::RoctxDataSource;
using rpdtracer::RoctxDataSourcePrivate;
using rpdtracer::ApiTable;
using rpdtracer::Logger;
using rpdtracer::GetPid;
using rpdtracer::GetTid;
using rpdtracer::clocktime_ns;
using rpdtracer::timestamp_t;

namespace rpdtracer {

class RoctxDataSourcePrivate
{
public:
    std::atomic<bool> active{false};

    sqlite3_int64 domainId{0};
    sqlite3_int64 rangeCategoryId{0};
    sqlite3_int64 markCategoryId{0};
    sqlite3_int64 apiNameId{0};

    // Timestamp of the most recent startTracing().  Set on the first start
    // too, so it is the trace start for a never-suspended run.  Used as the
    // left bound for ranges that were pushed before we were recording -- we
    // know they began before this, but not when.
    std::atomic<sqlite3_int64> resumeTime{0};

    bool idsCached{false};
    void cacheIds();
};

}    // namespace rpdtracer


extern "C" {
    DataSource *RoctxDataSourceFactory() { return new RoctxDataSource(); }
}

static RoctxDataSource *s_instance = nullptr;

RoctxDataSource &RoctxDataSource::instance()
{
    return *s_instance;
}

RoctxDataSource::RoctxDataSource()
: d(new RoctxDataSourcePrivate())
{
}

RoctxDataSource::~RoctxDataSource()
{
    delete d;
}

// Per-thread roctx range stack
static thread_local std::deque<ApiTable::row> t_roctxStack;

// Track all thread stacks for shutdown drain
static std::mutex s_stacksMutex;
static std::map<std::pair<int,int>, std::deque<ApiTable::row>*> s_stacks;

static void registerThreadStack()
{
    static thread_local bool registered = false;
    if (!registered) {
        std::lock_guard<std::mutex> lock(s_stacksMutex);
        auto key = std::pair<int,int>(GetPid(), GetTid());
        s_stacks[key] = &t_roctxStack;
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


void RoctxDataSourcePrivate::cacheIds()
{
    if (idsCached)
        return;
    Logger &logger = Logger::singleton();
    domainId = logger.stringTable().getOrCreate("roctx");
    rangeCategoryId = logger.stringTable().getOrCreate("range");
    markCategoryId = logger.stringTable().getOrCreate("mark");
    apiNameId = logger.stringTable().getOrCreate("UserMarker");
    idsCached = true;
}

// ---- roctx shim functions ----


extern "C" {

void roctxMarkA(const char *message)
{
    RoctxDataSourcePrivate *d = RoctxDataSource::instance().priv();
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

int roctxRangePushA(const char *message)
{
    RoctxDataSourcePrivate *d = RoctxDataSource::instance().priv();
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

    t_roctxStack.push_front(row);
    return static_cast<int>(t_roctxStack.size()) - 1;
}

int roctxRangePop()
{
    RoctxDataSourcePrivate *d = RoctxDataSource::instance().priv();
    if (!d->active.load(std::memory_order_relaxed))
        return -1;

    d->cacheIds();
    Logger &logger = Logger::singleton();

    // A pop with nothing on our stack means the app opened a range we never
    // saw -- either it was pushed while tracing was suspended (pushes are
    // squelched at the source, so no depth is tracked), or the app is
    // unbalanced.  Either way the pop is evidence that an enclosing frame
    // existed.  Dropping it would silently re-parent every descendant to
    // depth 0, since viewers derive nesting from start/end containment.
    // Emit a placeholder instead, clamped to resumeTime: we know the range
    // began before we started recording, but not when.
    if (t_roctxStack.empty()) {
        ApiTable::row row;
        row.pid = GetPid();
        row.tid = GetTid();
        row.start = d->resumeTime.load(std::memory_order_relaxed);
        row.end = clocktime_ns();
        row.domain_id = d->domainId;
        row.category_id = d->rangeCategoryId;
        row.apiName_id = d->apiNameId;
        row.args_id = logger.ustringTable().create("<pre-existing>");
        row.api_id = Logger::singleton().nextAnnotationId();

        logger.apiTable().insert(row);
        return -1;
    }

    ApiTable::row row = t_roctxStack.front();
    t_roctxStack.pop_front();

    row.end = clocktime_ns();
    row.api_id = Logger::singleton().nextAnnotationId();

    logger.apiTable().insert(row);
    return static_cast<int>(t_roctxStack.size());
}

}  // extern "C"


// ---- DataSource interface ----

void RoctxDataSource::init()
{
    s_instance = this;
}

void RoctxDataSource::startTracing()
{
    d->resumeTime.store(clocktime_ns(), std::memory_order_relaxed);
    d->active.store(true, std::memory_order_release);
}

// Ranges do not span a stop/start.  stopTracing() closes every open range at
// the stop timestamp, so a push/pop pair straddling a pause is recorded as a
// range ending at the stop.  The matching pop after the restart finds an
// empty stack and is recorded as a "<pre-existing>" range starting at the
// resume, preserving the nesting of anything opened after it.
void RoctxDataSource::stopTracing()
{
    d->active.store(false, std::memory_order_relaxed);

    // Drain in-flight ranges so their string ids (valid only in the
    // current storage) cannot outlive a resetStorage()
    drainStacks();
}

void RoctxDataSource::flush()
{
}

void RoctxDataSource::reset()
{
    d->idsCached = false;
}

void RoctxDataSource::end()
{
    d->active.store(false, std::memory_order_relaxed);
    drainStacks();
}
