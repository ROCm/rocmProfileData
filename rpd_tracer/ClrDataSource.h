/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"

#include <cstdint>
#include <vector>

namespace rpdtracer {

class ClrDataSource : public DataSource
{
public:
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;

private:
    std::vector<uint64_t> m_startIds;
    std::vector<uint64_t> m_endIds;
    size_t m_processedCount {0};
};

}    // namespace rpdtracer
