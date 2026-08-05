// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

//#include "Logger.h"

namespace rpdtracer {

class DataSource
{
public:
    virtual ~DataSource() = default;
    virtual void init() = 0;
    virtual void end() = 0;
    virtual void startTracing() = 0;
    virtual void stopTracing() = 0;
    virtual void flush() = 0;
};

}    // namespace rpdtracer
