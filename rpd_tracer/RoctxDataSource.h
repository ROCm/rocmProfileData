// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "DataSource.h"

namespace rpdtracer {

class RoctxDataSourcePrivate;
class RoctxDataSource : public DataSource
{
public:
    RoctxDataSource();
    ~RoctxDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;

    static RoctxDataSource &instance();
    RoctxDataSourcePrivate *priv() { return d; }

private:
    RoctxDataSourcePrivate *d;
    friend class RoctxDataSourcePrivate;
};

}    // namespace rpdtracer
