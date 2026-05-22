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
#include "Storage.h"

namespace rpdtracer {

const sqlite_int64 EMPTY_STRING_ID = 1;

class Logger
{
public:
    Logger() { init(); }
    static Logger& singleton();

    // Table writer classes.  Used directly by DataSources
    MetadataTable &metadataTable() { return m_storage->metadataTable(); }
    StringTable &stringTable() { return m_storage->stringTable(); }
    UStringTable &ustringTable() { return m_storage->ustringTable(); }
    OpTable &opTable() { return m_storage->opTable(); }
    KernelApiTable &kernelApiTable() { return m_storage->kernelApiTable(); }
    CopyApiTable &copyApiTable() { return m_storage->copyApiTable(); }
    ApiTable &apiTable() { return m_storage->apiTable(); }
    MonitorTable &monitorTable() { return m_storage->monitorTable(); }
    StackFrameTable &stackFrameTable() { return m_storage->stackFrameTable(); }

    // External control to stop/stop logging
    void rpdstart();
    void rpdstop();
    void rpdflush();

    // Insert an api event.  Used to log internal state or performance
    void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);


    // Used on library load and unload.
    //  Needs assistance from DataSources to avoid shutdown corruption
    static void rpdInit();
    static void rpdFinalize();

    const std::string filename() { return m_storage->filename(); };
    sqlite3_int64 nextAnnotationId() { return m_storage->nextAnnotationId(); }
    bool writeStackFrames() { return m_writeStackFrames; };

    void resetStorage();

private:
    int m_activeCount {0};
    std::mutex m_activeMutex;

    std::deque<DataSource*> m_sources;

    Storage *m_storage {nullptr};

    void init();
    void finalize();
    std::atomic<bool> m_writeOverheadRecords {true};
    bool m_writeStackFrames {false};

    std::atomic<bool> m_done {false};
    int m_period{1};
    std::thread *m_worker {nullptr};
    void autoflushWorker();
};

}    // namespace rpdtracer
