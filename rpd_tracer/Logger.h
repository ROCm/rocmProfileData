/*********************************************************************************
* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
********************************************************************************/
#pragma once

#include <string>
#include <mutex>
#include <deque>
#include <thread>

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
    OpTable *m_opTable {nullptr};
    KernelApiTable *m_kernelApiTable {nullptr};
    CopyApiTable *m_copyApiTable {nullptr};
    ApiTable *m_apiTable {nullptr};
    MonitorTable *m_monitorTable {nullptr};
    StackFrameTable *m_stackFrameTable {nullptr};

    void init();
    void finalize();

    std::string m_filename;
    bool m_writeOverheadRecords {true};
    bool m_writeStackFrames {false};

    bool m_done {false};
    int m_period{1};
    std::thread *m_worker {nullptr};
    void autoflushWorker();
};
