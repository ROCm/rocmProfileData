/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "DataSource.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <hip/amd_detail/hip_profiler_ext.h>

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
    static void chunkCallback(const hipApiRecordExt* records, uint32_t count,
                              uint32_t chunk_id, void* user_data);
    void processRecord(const hipApiRecordExt* r);

    class ApiStringList
    {
    public:
        ApiStringList() : m_invert(true) {}
        bool invertMode() { return m_invert; }
        void setInvertMode(bool invert) { m_invert = invert; }
        void add(const std::string &apiName) { m_filter.insert(apiName); }
        void remove(const std::string &apiName) { m_filter.erase(apiName); }
        bool loadUserPrefs() { return false; }
        bool contains(const std::string &apiName)
        {
            return (m_filter.find(apiName) != m_filter.end()) ? !m_invert : m_invert;
        }
    private:
        std::unordered_set<std::string> m_filter;
        bool m_invert;
    };

    struct Range {
        uint64_t start;
        uint64_t end;
    };

    ApiStringList m_apiList;
    std::vector<Range> m_ranges;
    std::atomic<uint64_t> m_correlationId {0};
    size_t m_processedCount {0};
};

}    // namespace rpdtracer
