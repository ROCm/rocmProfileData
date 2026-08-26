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
#include "RlogDataSource.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <map>

#include <sqlite3.h>

#include "Logger.h"
#include "Utility.h"

#include "rlog/Hub.h"

using rpdtracer::DataSource;
using rpdtracer::RlogDataSource;
using rpdtracer::RlogDataSourcePrivate;
using rpdtracer::ApiTable;
using rpdtracer::Logger;
using rpdtracer::GetPid;
using rpdtracer::GetTid;
using rpdtracer::clocktime_ns;
using rpdtracer::timestamp_t;
using rpdtracer::EMPTY_STRING_ID;

namespace rpdtracer {

class RlogDataSourcePrivate
{
public:
};

}    // namespace rpdtracer


extern "C" {
    DataSource *RlogDataSourceFactory() { return new RlogDataSource(); }
}

RlogDataSource::RlogDataSource()
: d(new RlogDataSourcePrivate())
{
}

RlogDataSource::~RlogDataSource()
{
    delete d;
}

// Per-thread rlog range stack
static thread_local std::deque<ApiTable::row> t_rlogStack;

// Track all thread stacks for suspend drain
static std::mutex s_stacksMutex;
static std::map<std::pair<int,int>, std::deque<ApiTable::row>*> s_stacks;

static void registerThreadStack()
{
    static thread_local bool registered = false;
    if (!registered) {
        std::lock_guard<std::mutex> lock(s_stacksMutex);
        auto key = std::pair<int,int>(GetPid(), GetTid());
        s_stacks[key] = &t_rlogStack;
        registered = true;
    }
}


void RlogDataSource::init()
{
}

void RlogDataSource::startTracing()
{
    rlog::Hub::singleton().addLogger(*this);
}

void RlogDataSource::stopTracing()
{
    rlog::Hub::singleton().removeLogger(*this);

    // No more callbacks will arrive. Drain all in-flight ranges.
    timestamp_t now = clocktime_ns();
    rpdtracer::Logger &logger = rpdtracer::Logger::singleton();

    std::lock_guard<std::mutex> lock(s_stacksMutex);
    for (auto &entry : s_stacks) {
        auto &stack = *entry.second;
        while (!stack.empty()) {
            ApiTable::row row = stack.front();
            stack.pop_front();
            row.end = now;
            row.api_id = rpdtracer::Logger::singleton().nextAnnotationId();
            logger.apiTable().insert(row);
        }
    }
}

void RlogDataSource::flush()
{
}

void RlogDataSource::end()
{
    stopTracing();
}


void RlogDataSource::mark(const char *domain, const char *category, const char *apiname, const char *args)
{
    rpdtracer::Logger &logger = rpdtracer::Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    row.domain_id = logger.stringTable().getOrCreate(std::string(domain));
    row.category_id = logger.stringTable().getOrCreate(std::string(category));
    row.apiName_id = logger.stringTable().getOrCreate(std::string(apiname));
    row.args_id = logger.ustringTable().create(args);
    row.api_id = rpdtracer::Logger::singleton().nextAnnotationId();

    logger.apiTable().insert(row);
}

void RlogDataSource::rangePush(const char *domain, const char *category, const char *apiname, const char *args)
{
    registerThreadStack();

    rpdtracer::Logger &logger = rpdtracer::Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = 0;
    row.domain_id = logger.stringTable().getOrCreate(std::string(domain));
    row.category_id = logger.stringTable().getOrCreate(std::string(category));
    row.apiName_id = logger.stringTable().getOrCreate(std::string(apiname));
    row.args_id = logger.ustringTable().create(args);
    row.api_id = 0;

    t_rlogStack.push_front(row);
}

void RlogDataSource::rangePop()
{
    if (t_rlogStack.empty())
        return;

    rpdtracer::Logger &logger = rpdtracer::Logger::singleton();

    ApiTable::row row = t_rlogStack.front();
    t_rlogStack.pop_front();

    row.end = clocktime_ns();
    row.api_id = rpdtracer::Logger::singleton().nextAnnotationId();

    logger.apiTable().insert(row);
}
