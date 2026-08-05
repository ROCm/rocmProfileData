// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "DataSource.h"

namespace rpdtracer {

class NvtxDataSourcePrivate;
class NvtxDataSource : public DataSource
{
public:
    NvtxDataSource();
    ~NvtxDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;

    static NvtxDataSource &instance();
    NvtxDataSourcePrivate *priv() { return d; }

private:
    NvtxDataSourcePrivate *d;
    friend class NvtxDataSourcePrivate;
};

}    // namespace rpdtracer
