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
#include "Logger.h"

#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "Utility.h"


#if 0
static void rpdInit() __attribute__((constructor));
static void rpdFinalize() __attribute__((destructor));
// FIXME: can we avoid shutdown corruption?
// Other rocm libraries crashing on unload
// libsqlite unloading before we are done using it
// Current workaround: register an onexit function when first activity is delivered back
//                     this let's us unload first, or close to.
// New workaround: register 3 times, only finalize once.  see register_once

std::once_flag register_once;
std::once_flag registerAgain_once;
#endif


// Hide the C-api here for now
extern "C" {
void rpdstart()
{
    Logger::singleton().rpdstart();
}

void rpdstop()
{
    Logger::singleton().rpdstop();
}

void rpdflush()
{
    Logger::singleton().rpdflush();
}

void rpd_rangePush(const char *domain, const char *apiName, const char* args)
{
    Logger::singleton().rpd_rangePush(domain, apiName, args);
}

void rpd_rangePop()
{
    Logger::singleton().rpd_rangePop();
}
}  // extern "C"

// GFH - This mirrors the function in the pre-refactor code.  Allows both code paths to compile.
//   See table classes for users.  Todo: build a proper threaded record writer
void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args)
{
    Logger::singleton().createOverheadRecord(start, end, name, args);
}


Logger& Logger::singleton()
{
    static Logger logger;
    return logger;
}

void Logger::rpdInit() {
    Logger::singleton().init();
}

void Logger::rpdFinalize() {
    Logger::singleton().finalize();
}


void Logger::rpdstart()
{
    std::unique_lock<std::mutex> lock(m_activeMutex);
    if (m_activeCount == 0) {
        //fprintf(stderr, "rpd_tracer: START\n");
        m_apiTable->resumeRoctx(clocktime_ns());
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->startTracing();
    }
    ++m_activeCount;
}

void Logger::rpdstop()
{
    std::unique_lock<std::mutex> lock(m_activeMutex);
    if (m_activeCount == 1) {
        //fprintf(stderr, "rpd_tracer: STOP\n");
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->stopTracing();
        m_apiTable->suspendRoctx(clocktime_ns());
    }
    --m_activeCount;
}

void Logger::rpdflush()
{
    std::unique_lock<std::mutex> lock(m_activeMutex);
    //fprintf(stderr, "rpd_tracer: FLUSH\n");
    const timestamp_t cb_begin_time = clocktime_ns();

    // Have the data sources flush out whatever they have available
    for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->flush();

    m_stringTable->flush();
    m_kernelApiTable->flush();
    m_copyApiTable->flush();
    m_opTable->flush();
    m_apiTable->flush();
    m_monitorTable->flush();
    m_stackFrameTable->flush();

    const timestamp_t cb_end_time = clocktime_ns();
    createOverheadRecord(cb_begin_time, cb_end_time, "rpdflush", "");
}

void Logger::rpd_rangePush(const char *domain, const char *apiName, const char* args)
{
    {
        std::unique_lock<std::mutex> lock(m_activeMutex);
        if (m_activeCount == 0)
            return;
    }
    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    row.apiName_id = m_stringTable->getOrCreate(apiName);
    row.args_id = m_stringTable->getOrCreate(args);
    row.api_id = 0;
    m_apiTable->pushRoctx(row);
}

void Logger::rpd_rangePop()
{
    {
        std::unique_lock<std::mutex> lock(m_activeMutex);
        if (m_activeCount == 0)
            return;
    }
    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    row.apiName_id = EMPTY_STRING_ID;
    row.args_id = EMPTY_STRING_ID;
    row.api_id = 0;
    m_apiTable->popRoctx(row);
}




void Logger::init()
{
    fprintf(stderr, "rpd_tracer, because\n");

    const char *filename = getenv("RPDT_FILENAME");
    if (filename == NULL)
        filename = "./trace.rpd";
    m_filename = filename;

    // Indicate the tracer loaded.  Used for snooping without loading
    setenv("RPDT_LOADED", "1", 1);

    // Create table recorders

    m_metadataTable = new MetadataTable(filename);
    m_stringTable = new StringTable(filename);
    m_kernelApiTable = new KernelApiTable(filename);
    m_copyApiTable = new CopyApiTable(filename);
    m_opTable = new OpTable(filename);
    m_apiTable = new ApiTable(filename);
    m_monitorTable = new MonitorTable(filename);
    m_stackFrameTable = new StackFrameTable(filename);

    // Offset primary keys so they do not collide between sessions
    sqlite3_int64 offset = m_metadataTable->sessionId() * (sqlite3_int64(1) << 32);
    m_metadataTable->setIdOffset(offset);
    m_stringTable->setIdOffset(offset);
    m_kernelApiTable->setIdOffset(offset);
    m_copyApiTable->setIdOffset(offset);
    m_opTable->setIdOffset(offset);
    m_apiTable->setIdOffset(offset);
    m_stackFrameTable->setIdOffset(offset);

    // Create one instance of each available datasource
    std::list<std::string> factories = {
        "RoctracerDataSourceFactory",
        "CuptiDataSourceFactory",
        "RocmSmiDataSourceFactory"
        };

    void (*dl) = dlopen("librpd_tracer.so", RTLD_LAZY);
    if (dl) {
        for (auto it = factories.begin(); it != factories.end(); ++it) {
            DataSource* (*func) (void) = (DataSource* (*)()) dlsym(dl, (*it).c_str());
            if (func) {
                m_sources.push_back(func());
                //fprintf(stderr, "Using: %s\n", (*it).c_str());
            }
        }
    }

    // Initialize data sources
    for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->init();

    // Allow starting with recording disabled via ENV
    bool startTracing = true;
    char *val = getenv("RPDT_AUTOSTART");
    if (val != NULL) {
        int autostart = atoi(val);
        if (autostart == 0)
            startTracing = false;
    }
    if (startTracing == true) {
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->startTracing();
        std::unique_lock<std::mutex> lock(m_activeMutex);
        ++m_activeCount;
    }
    static std::once_flag register_once;
    std::call_once(register_once, atexit, Logger::rpdFinalize);

    // Start autoflush hack
    const char *autoflush = getenv("RPDT_AUTOFLUSH");
    if (autoflush != nullptr) {
        int frequency = atoi(autoflush);
        if (frequency > 0) {
            m_period = 1000000 / frequency;  // usecs
            m_done = false;
            m_worker = new std::thread(&Logger::autoflushWorker, this);
        }
    }

    // Enable stack frame recording
    const char *stackframe = getenv("RPDT_STACKFRAMES");
    if (stackframe != nullptr) {
        int val = atoi(stackframe);
        m_writeStackFrames = (val != 0);
    }
}

static bool doFinalize = true;
std::mutex finalizeMutex;

void Logger::finalize()
{
    std::lock_guard<std::mutex> guard(finalizeMutex);
    if (doFinalize == true) {
        doFinalize = false;

        m_done = true;
        if (m_worker != nullptr)
            m_worker->join();	// deadlock in here.  try skipping if needed

        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->stopTracing();

        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->end();

        // Flush recorders
        const timestamp_t begin_time = clocktime_ns();
        m_opTable->finalize();		// OpTable before subclassOpTables
        m_kernelApiTable->finalize();
        m_copyApiTable->finalize();
        m_monitorTable->finalize();
        m_stackFrameTable->finalize();
        m_writeOverheadRecords = false;	// Don't make any new overhead records (api calls)
        m_apiTable->finalize();
        m_stringTable->finalize();	// String table last

        const timestamp_t end_time = clocktime_ns();
        fprintf(stderr, "rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);
    }
}

void Logger::autoflushWorker()
{
    while (m_done == false) {
        rpdflush();
        usleep(1000000);
    }
}

void Logger::createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args)
{
    if (m_writeOverheadRecords == false)
        return;
    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = start;
    row.end = end;
    row.apiName_id = m_stringTable->getOrCreate(name);
    row.args_id = m_stringTable->getOrCreate(args);
    row.api_id = 0;

    //fprintf(stderr, "overhead: %s (%s) - %f usec\n", name.c_str(), args.c_str(), (end-start) / 1000.0);

    m_apiTable->insertRoctx(row);
}

