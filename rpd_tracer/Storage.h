/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <atomic>
#include <string>
#include <sqlite3.h>
#include "Table.h"

namespace rpdtracer {

class Storage
{
public:
    Storage(const char *filename, bool directWrite);
    ~Storage();

    MetadataTable &metadataTable() { return *m_metadataTable; }
    StringTable &stringTable() { return *m_stringTable; }
    UStringTable &ustringTable() { return *m_ustringTable; }
    KernelApiTable &kernelApiTable() { return *m_kernelApiTable; }
    CopyApiTable &copyApiTable() { return *m_copyApiTable; }
    OpTable &opTable() { return *m_opTable; }
    ApiTable &apiTable() { return *m_apiTable; }
    MonitorTable &monitorTable() { return *m_monitorTable; }
    StackFrameTable &stackFrameTable() { return *m_stackFrameTable; }

    void flush();
    void finalize();

    const std::string &filename() const { return m_filename; }
    sqlite3_int64 sessionId() const;

    sqlite3_int64 overheadDomainId();
    sqlite3_int64 overheadCategoryId();

    sqlite3_int64 nextAnnotationId() { return m_annotationIdCounter.fetch_add(1, std::memory_order_relaxed); }

private:
    std::string m_filename;
    MetadataTable *m_metadataTable {nullptr};
    StringTable *m_stringTable {nullptr};
    UStringTable *m_ustringTable {nullptr};
    KernelApiTable *m_kernelApiTable {nullptr};
    CopyApiTable *m_copyApiTable {nullptr};
    OpTable *m_opTable {nullptr};
    ApiTable *m_apiTable {nullptr};
    MonitorTable *m_monitorTable {nullptr};
    StackFrameTable *m_stackFrameTable {nullptr};

    sqlite3_int64 m_overheadDomainId {0};
    sqlite3_int64 m_overheadCategoryId {0};
    bool m_overheadIdsCached {false};
    std::atomic<sqlite3_int64> m_annotationIdCounter{sqlite3_int64(1) << 31};
    bool m_finalized {false};
};

}    // namespace rpdtracer
