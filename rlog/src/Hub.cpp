// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <cassert>

#include "Hub.h"
#include "PropertyDb.h"

using namespace rlog;

extern "C" {
void rlog_mark(const char *domain, const char *category, const char *apiName, const char* args)
{
    Hub::singleton().mark(domain, category, apiName, args);
}

void rlog_rangePush(const char *domain, const char *category, const char *apiName, const char* args)
{
    Hub::singleton().rangePush(domain, category, apiName, args);
}

void rlog_rangePop()
{
    Hub::singleton().rangePop();
}

const char* rlog_getProperty(const char *domain, const char *property, const char *defaultValue)
{
    return Hub::singleton().getProperty(domain, property, defaultValue);
}

void rlog_registerActiveCallback(void (*cb)())
{
    Hub::singleton().registerActiveCallback(cb);
}

bool rlog_isActive()
{
    return Hub::singleton().isActive();
}
}

namespace rlog {

class HubPrivate
{
public:
    std::unordered_map<Logger*, int> loggers;
    std::unordered_map<void(*)(), int> callbacks;

    std::atomic<bool> active{false};
    std::atomic<Logger*> singleLogger{nullptr};

    std::mutex mutex;

    std::unique_ptr<PropertyDb> propertyDb;

    void notifyListeners();
};

Hub::Hub()
: d(std::make_unique<HubPrivate>())
{
    try {
        d->propertyDb = std::make_unique<PropertyDb>();
    } catch (const std::exception&) {
        d->propertyDb = nullptr;
    }
}

Hub::~Hub() = default;

Hub& Hub::singleton()
{
    static Hub logger;
    return logger;
}



void Hub::mark(const char *domain, const char *category, const char *apiName, const char* args)
{
    Logger* single = d->singleLogger.load(std::memory_order_acquire);
    if (single) { single->mark(domain, category, apiName, args); return; }
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->mark(domain, category, apiName, args);
    }
}

void Hub::rangePush(const char *domain, const char *category, const char *apiName, const char* args)
{
    Logger* single = d->singleLogger.load(std::memory_order_acquire);
    if (single) { single->rangePush(domain, category, apiName, args); return; }
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->rangePush(domain, category, apiName, args);
    }
}

void Hub::rangePop()
{
    Logger* single = d->singleLogger.load(std::memory_order_acquire);
    if (single) { single->rangePop(); return; }
    std::unique_lock<std::mutex> lock(d->mutex);
    for (const auto &it: d->loggers) {
       (it.first)->rangePop();
    }
}

void Hub::addLogger(Logger &logger)
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
    d->singleLogger.store(postsize == 1 ? d->loggers.begin()->first : nullptr,
                          std::memory_order_release);
    if (postsize > 0 and presize < 1) {
        d->active = true;
        d->notifyListeners();
    }
}

void Hub::removeLogger(Logger &logger)
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
    d->singleLogger.store(postsize == 1 ? d->loggers.begin()->first : nullptr,
                          std::memory_order_release);
    if (postsize <=0 and presize >= 1) {
        d->active = false;
        d->notifyListeners();
    }
}

const char *Hub::getProperty(const char *domain, const char *property, const char *defaultValue)
{
    if (d->propertyDb)
        return d->propertyDb->getProperty(domain, property, defaultValue);
    return defaultValue;
}

void Hub::registerActiveCallback(void (*cb)())
{
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

bool Hub::isActive()
{
    return d->active;
}

void HubPrivate::notifyListeners()
{
    assert(this->mutex.try_lock() == false && "Should be holding lock");
    for (const auto &it: this->callbacks) {
       it.first();
    }
}

} // namespace rlog
