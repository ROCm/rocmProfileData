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

#include <atomic>
#include <map>
#include <mutex>
#include <cassert>

#include "RLogger.h"

using namespace rlog;

extern "C" {
void rlog_mark(const char *domain, const char *category, const char *apiName, const char* args)
{
    RLogger::singleton().mark(domain, category, apiName, args);
}

void rlog_rangePush(const char *domain, const char *category, const char *apiName, const char* args)
{
    RLogger::singleton().rangePush(domain, category, apiName, args);
}

void rlog_rangePop()
{
    RLogger::singleton().rangePop();
}

void rlog_getProperty(const char *domain, const char *property, const char *defaultValue)
{
    RLogger::singleton().getProperty(domain, property, defaultValue);
}

void rlog_registerActiveCallback(void (*cb)())
{
    RLogger::singleton().registerActiveCallback(cb);
}

bool rlog_isActive()
{
    return RLogger::singleton().isActive();
}
}

namespace rlog {

class RLoggerPrivate
{
public:
    //RLoggerPrivate();

    std::map<Logger*, int> loggers;
    std::map<void(*)(), int> callbacks;

    std::atomic<bool> active{false};

    std::mutex mutex;

    void notifyListeners();
};

RLogger::RLogger()
: d(new RLoggerPrivate())
{
}

RLogger::~RLogger()
{
    delete d;
}

RLogger& RLogger::singleton()
{
    static RLogger logger;
    return logger;
}


void RLogger::rlogInit() {
    RLogger::singleton().init();
}

void RLogger::rlogFinalize() {
    RLogger::singleton().finalize();
}

void RLogger::init()
{
    fprintf(stderr, "RLogger::init\n");
}

void RLogger::finalize()
{
    fprintf(stderr, "RLogger::finalize\n");
}

void RLogger::mark(const char *domain, const char *category, const char *apiName, const char* args)
{
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->mark(domain, category, apiName, args);
    }
    fprintf(stderr, "MARK\n");
}

void RLogger::rangePush(const char *domain, const char *category, const char *apiName, const char* args)
{
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->rangePush(domain, category, apiName, args);
    }
    fprintf(stderr, "PUSH\n");
}

void RLogger::rangePop()
{
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->rangePop();
    }
    fprintf(stderr, "POP\n");
}

void RLogger::addLogger(Logger &logger)
{
    std::unique_lock<std::mutex> lock(d->mutex);
    int presize = d->loggers.size();
    auto it = d->loggers.find(&logger);
    if (it == d->loggers.end()) {
        d->loggers.insert({&logger, 1});
    }
    else {
        ++(it->second);
    }
    int postsize = d->loggers.size();
    if (postsize > 0 and presize < 1) {
        d->active = true;
        d->notifyListeners();
    }
    fprintf(stderr, "addLogger\n");
}

void RLogger::removeLogger(Logger &logger)
{
    std::unique_lock<std::mutex> lock(d->mutex);
    int presize = d->loggers.size();
    auto it = d->loggers.find(&logger);
    if (it != d->loggers.end()) {
        --(it->second);
        if (it->second <= 0)
            it = d->loggers.erase(it);
    }
    int postsize = d->loggers.size();
    if (postsize <=0 and presize >= 1) {
        d->active = false;
        d->notifyListeners();
    }
    fprintf(stderr, "removeLogger\n");
}

const char *RLogger::getProperty(const char *domain, const char *property, const char *defaultValue)
{
    fprintf(stderr, "getProperty\n");
    return defaultValue;
}

void RLogger::registerActiveCallback(void (*cb)())
{
    fprintf(stderr, "RegisterActiveCallback\n");

    std::unique_lock<std::mutex> lock(d->mutex);
    auto it = d->callbacks.find(cb);
    if (it == d->callbacks.end()) {
        d->callbacks.insert({cb, 1});
    }
    else {
        ++(it->second);
    }
    cb();
}

bool RLogger::isActive()
{
    return d->active;
}

void RLoggerPrivate::notifyListeners()
{
    assert(this->mutex.try_lock() == false && "Should be holding lock");
    for (const auto &it: this->callbacks) {
       it.first();
    }
}

} // namespace rlog
