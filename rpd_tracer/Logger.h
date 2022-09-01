/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <deque>

#include "Table.h"
#include "DataSource.h"

const sqlite_int64 EMPTY_STRING_ID = 1;

class Logger
{
public:
    //Logger();
    static Logger& singleton();

    // Table writer classes.  Used directly by DataSources
    MetadataTable &metadataTable() { return *m_metadataTable; }
    StringTable &stringTable() { return *m_stringTable; }
    OpTable &opTable() { return *m_opTable; }
    KernelApiTable &kernelApiTable() { return *m_kernelApiTable; }
    CopyApiTable &copyApiTable() { return *m_copyApiTable; }
    ApiTable &apiTable() { return *m_apiTable; }


    // External control to stop/stop logging
    void rpdstart();
    void rpdstop();

    // Insert an api event.  Used to log internal state or performance
    void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);


    // Used on library load and unload.
    //  Needs assistance from DataSources to avoid shutdown corruption
    static void rpdInit() __attribute__((constructor));
    static void rpdFinalize() __attribute__((destructor));

private:
    int m_activeCount {0};
    std::mutex m_activeMutex;

    std::deque<DataSource*> m_sources;

    MetadataTable *m_metadataTable {nullptr};
    StringTable *m_stringTable {nullptr};
    OpTable *m_opTable {nullptr};
    KernelApiTable *m_kernelApiTable {nullptr};
    CopyApiTable *m_copyApiTable {nullptr};
    ApiTable *m_apiTable {nullptr};

    void init();
    void finalize();
};
