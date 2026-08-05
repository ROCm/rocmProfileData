// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <deque>
#include <thread>

#include "Table.h"
#include "DataSource.h"

namespace rpdtracer {

const sqlite_int64 EMPTY_STRING_ID = 1;

class Logger
{
public:
    Logger() { init(); }
    static Logger& singleton();

    // Table writer classes.  Used directly by DataSources
    MetadataTable &metadataTable() { return *m_metadataTable; }
    StringTable &stringTable() { return *m_stringTable; }
    UStringTable &ustringTable() { return *m_ustringTable; }
    OpTable &opTable() { return *m_opTable; }
    KernelApiTable &kernelApiTable() { return *m_kernelApiTable; }
    CopyApiTable &copyApiTable() { return *m_copyApiTable; }
    ApiTable &apiTable() { return *m_apiTable; }
    MonitorTable &monitorTable() { return *m_monitorTable; }
    StackFrameTable &stackFrameTable() { return *m_stackFrameTable; }

    // External control to stop/stop logging
    void rpdstart();
    void rpdstop();
    void rpdflush();

    // External maker api
    void rpd_rangePush(const char *domain, const char *apiName, const char* args);
    void rpd_rangePop();

    // Insert an api event.  Used to log internal state or performance
    void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);


    // Used on library load and unload.
    //  Needs assistance from DataSources to avoid shutdown corruption
    static void rpdInit() __attribute__((constructor));
    static void rpdFinalize() __attribute__((destructor));

    const std::string filename() { return m_filename; };
    bool writeStackFrames() { return m_writeStackFrames; };

private:
    int m_activeCount {0};
    std::mutex m_activeMutex;

    std::deque<DataSource*> m_sources;

    MetadataTable *m_metadataTable {nullptr};
    StringTable *m_stringTable {nullptr};
    UStringTable *m_ustringTable {nullptr};
    OpTable *m_opTable {nullptr};
    KernelApiTable *m_kernelApiTable {nullptr};
    CopyApiTable *m_copyApiTable {nullptr};
    ApiTable *m_apiTable {nullptr};
    MonitorTable *m_monitorTable {nullptr};
    StackFrameTable *m_stackFrameTable {nullptr};

    void init();
    void finalize();

    std::string m_filename;
    std::atomic<bool> m_writeOverheadRecords {true};
    bool m_writeStackFrames {false};

    std::atomic<bool> m_done {false};
    int m_period{1};
    std::thread *m_worker {nullptr};
    void autoflushWorker();
};

}    // namespace rpdtracer
