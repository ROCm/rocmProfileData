// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rlog/Logger.h"
#include <memory>

namespace rlog {

class HubPrivate;
class Hub
{
public:
    static Hub& singleton();

    // Mirrors Logger interface; fans out to all registered loggers
    void mark(const char *domain, const char *category, const char *apiName, const char* args);
    void rangePush(const char *domain, const char *category, const char *apiName, const char* args);
    void rangePop();

    // Add or remove a logger - ref counted per logger
    void addLogger(Logger &logger);
    void removeLogger(Logger &logger);

    // Active is true when any logger in present
    void registerActiveCallback(void (*cb)());
    bool isActive();

    // Properties
    const char *getProperty(const char *domain, const char *property, const char *defaultValue);

private:
    Hub();
    ~Hub();

    std::unique_ptr<HubPrivate> d;
    friend class HubPrivate;

};

}  // namespace rlog
