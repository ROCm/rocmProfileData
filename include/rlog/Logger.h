// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

namespace rlog {

class Logger {
public:
    virtual ~Logger() = default;
    virtual void mark(const char *domain, const char *category, const char *apiname, const char *args) = 0;
    virtual void rangePush(const char *domain, const char *category, const char *apiname, const char *args) = 0;
    virtual void rangePop() = 0;
};


}  // namespace rlog
