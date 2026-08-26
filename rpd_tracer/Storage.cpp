/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Storage.h"

#include <stdio.h>
#include <fmt/format.h>

#include "Utility.h"
#include "Schema.h"

using rpdtracer::Storage;

Storage::Storage(const char *filename, bool directWrite)
: m_filename(filename)
{
    ensureSchema(filename);

    m_metadataTable = new MetadataTable(filename);
    m_stringTable = new StringTable(filename, directWrite);
    m_ustringTable = new UStringTable(filename, directWrite);
    m_kernelApiTable = new KernelApiTable(filename, directWrite);
    m_copyApiTable = new CopyApiTable(filename, directWrite);
    m_opTable = new OpTable(filename, directWrite);
    m_apiTable = new ApiTable(filename, directWrite);
    m_monitorTable = new MonitorTable(filename, directWrite);
    m_stackFrameTable = new StackFrameTable(filename, directWrite);

    m_metadataTable->insert("session", fmt::format("id={} pid={}", m_metadataTable->sessionId(), GetPid()));

    sqlite3_int64 offset = m_metadataTable->sessionId() * (sqlite3_int64(1) << 32);
    m_metadataTable->setIdOffset(offset);
    m_stringTable->setIdOffset(offset);
    m_ustringTable->setIdOffset(offset);
    m_kernelApiTable->setIdOffset(offset);
    m_copyApiTable->setIdOffset(offset);
    m_opTable->setIdOffset(offset);
    m_apiTable->setIdOffset(offset);
    m_stackFrameTable->setIdOffset(offset);
}

Storage::~Storage()
{
    finalize();
    delete m_metadataTable;
    delete m_stringTable;
    delete m_ustringTable;
    delete m_kernelApiTable;
    delete m_copyApiTable;
    delete m_opTable;
    delete m_apiTable;
    delete m_monitorTable;
    delete m_stackFrameTable;
}

void Storage::flush()
{
    m_stringTable->flush();
    m_ustringTable->flush();
    m_kernelApiTable->flush();
    m_copyApiTable->flush();
    m_opTable->flush();
    m_apiTable->flush();
    m_monitorTable->flush();
    m_stackFrameTable->flush();
}

void Storage::finalize()
{
    if (m_finalized)
        return;
    m_finalized = true;

    const timestamp_t begin_time = clocktime_ns();
    m_opTable->finalize();
    m_kernelApiTable->finalize();
    m_copyApiTable->finalize();
    m_monitorTable->finalize();
    m_stackFrameTable->finalize();
    m_apiTable->finalize();
    m_ustringTable->finalize();
    m_stringTable->finalize();

    const timestamp_t end_time = clocktime_ns();
    fprintf(stderr, "rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);
}

sqlite3_int64 Storage::sessionId() const
{
    return m_metadataTable->sessionId();
}

sqlite3_int64 Storage::overheadDomainId()
{
    if (!m_overheadIdsCached) {
        m_overheadDomainId = m_stringTable->getOrCreate("rpd_tracer");
        m_overheadCategoryId = m_stringTable->getOrCreate("overhead");
        m_overheadIdsCached = true;
    }
    return m_overheadDomainId;
}

sqlite3_int64 Storage::overheadCategoryId()
{
    if (!m_overheadIdsCached) {
        m_overheadDomainId = m_stringTable->getOrCreate("rpd_tracer");
        m_overheadCategoryId = m_stringTable->getOrCreate("overhead");
        m_overheadIdsCached = true;
    }
    return m_overheadCategoryId;
}
