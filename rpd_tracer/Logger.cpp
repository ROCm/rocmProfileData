// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "Logger.h"

#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "Utility.h"
#include "Schema.h"

using rpdtracer::Logger;

namespace rpdtracer { void rlogClientInit(); }

// GFH - This mirrors the function in the pre-refactor code.  Allows both code paths to compile.
//   See table classes for users.  Todo: build a proper threaded record writer
void rpdtracer::createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args)
{
    Logger::singleton().createOverheadRecord(start, end, name, args);
}

namespace {
    bool loggerInitialized { false };
}

Logger& Logger::singleton()
{
    static Logger logger;
    return logger;
}

void Logger::rpdInit() {
    bool doInit = true;
    char *val = getenv("RPDT_DELAYINIT");
    if (val != NULL) {
        int delayinit = atoi(val);
        if (delayinit != 0)
            doInit = false;
    }
    if (doInit)
        Logger::singleton();

    // Indicate the tracer loaded.  Used for snooping without loading
    setenv("RPDT_LOADED", "1", 1);
}

void Logger::rpdFinalize() {
    if (loggerInitialized)
        Logger::singleton().finalize();
}

void Logger::resetStorage()
{
    m_storage->finalize();
    delete m_storage;
    const char *filename = getConfig("RPDT_FILENAME", "filename", "./trace.rpd");
    bool directWrite = (atoi(getConfig("RPDT_DIRECTWRITE", "directwrite", "0")) != 0);
    m_storage = new Storage(filename, directWrite);

    for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
        (*it)->reset();
}

void Logger::rpdstart()
{
    std::unique_lock<std::mutex> lock(m_activeMutex);
    if (m_activeCount == 0) {
        rlog::mark("rpd_tracer", "", "rpdstart", "");
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->startTracing();
    }
    ++m_activeCount;
}

void Logger::rpdstop()
{
    std::unique_lock<std::mutex> lock(m_activeMutex);
    if (m_activeCount == 1) {
        rlog::mark("rpd_tracer", "", "rpdstop", "");
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->stopTracing();
    }
    --m_activeCount;
}

void Logger::rpdflush()
{
    //fprintf(stderr, "rpd_tracer: FLUSH\n");
    const timestamp_t cb_begin_time = clocktime_ns();

    // Have the data sources flush out whatever they have available
    for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->flush();

    m_storage->flush();

    const timestamp_t cb_end_time = clocktime_ns();
    createOverheadRecord(cb_begin_time, cb_end_time, "rpdflush", "");
}





void Logger::init()
{
    fprintf(stderr, "rpd_tracer, because\n");

    rlogClientInit();

    rlog::getProperty("rpd_tracer", "filename", "./trace.rpd");
    const char *filename = getConfig("RPDT_FILENAME", "filename", "./trace.rpd");
    bool directWrite = false;

    const char *dwrite = getenv("RPDT_DIRECTWRITE");
    if (dwrite != nullptr) {
        int val = atoi(dwrite);
        directWrite = (val != 0);
    }

    m_storage = new Storage(filename, directWrite);

    // Create one instance of each available datasource
    std::list<std::string> factories = {
        "RoctxDataSourceFactory",
        "NvtxDataSourceFactory",
        "RocprofDataSourceFactory",
        "RoctracerDataSourceFactory",
        "CuptiDataSourceFactory",
        "RlogDataSourceFactory",
        "RocmSmiDataSourceFactory"
        };


    for (auto it = factories.begin(); it != factories.end(); ++it) {
        DataSource* (*func) (void) = (DataSource* (*)()) dlsym(RTLD_DEFAULT, (*it).c_str());
        if (func) {
            m_sources.push_back(func());
            //fprintf(stderr, "Using: %s\n", (*it).c_str());
        }
    }

    // Initialize data sources
    for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->init();

    // Allow starting with recording disabled via ENV
    bool startTracing = true;
    if (atoi(getConfig("RPDT_AUTOSTART", "autostart", "1")) == 0)
        startTracing = false;
    if (startTracing == true) {
        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->startTracing();
        std::unique_lock<std::mutex> lock(m_activeMutex);
        ++m_activeCount;
    }
    // Start autoflush hack
    {
        int frequency = atoi(getConfig("RPDT_AUTOFLUSH", "autoflush", "0"));
        if (frequency > 0) {
            m_period = 1000000 / frequency;  // usecs
            m_done = false;
            m_worker = new std::thread(&Logger::autoflushWorker, this);
        }
    }

    // Enable stack frame recording
    m_writeStackFrames = (atoi(getConfig("RPDT_STACKFRAMES", "stackframes", "0")) != 0);

    loggerInitialized = true;  // detect lazy init
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
            m_worker->join();

        {
            std::unique_lock<std::mutex> lock(m_activeMutex);
            if (m_activeCount > 0) {
                for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
                    (*it)->stopTracing();
            }
        }

        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->end();

        m_writeOverheadRecords = false;
        m_storage->finalize();
    }
}

void Logger::autoflushWorker()
{
    while (m_done == false) {
        rpdflush();
        usleep(m_period);
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
    row.domain_id = m_storage->overheadDomainId();
    row.category_id = m_storage->overheadCategoryId();
    row.apiName_id = m_storage->stringTable().getOrCreate(name);
    row.args_id = m_storage->ustringTable().create(args);
    row.api_id = m_storage->nextAnnotationId();

    m_storage->apiTable().insert(row);
}

