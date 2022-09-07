/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Logger.h"

//#include "hsa_rsrc_factory.h"

#include "Utility.h"

// FIMXE: remove and use static init
//#include "RoctracerDataSource.h"
#include "CuptiDataSource.h"


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
        //m_apiTable->resumeRoctx(util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC));
	//FIXME
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
        //m_apiTable->suspendRoctx(util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC));
	//FIXME
        m_apiTable->suspendRoctx(clocktime_ns());
    }
    --m_activeCount;
}




void Logger::init()
{
    fprintf(stderr, "rpd_tracer, because\n");

    const char *filename = getenv("RPDT_FILENAME");
    if (filename == NULL)
        filename = "./trace.rpd";

    // Create table recorders

    m_metadataTable = new MetadataTable(filename);
    m_stringTable = new StringTable(filename);
    m_kernelApiTable = new KernelApiTable(filename);
    m_copyApiTable = new CopyApiTable(filename);
    m_opTable = new OpTable(filename);
    m_apiTable = new ApiTable(filename);

    // Offset primary keys so they do not collide between sessions
    sqlite3_int64 offset = m_metadataTable->sessionId() * (sqlite3_int64(1) << 32);
    m_metadataTable->setIdOffset(offset);
    m_stringTable->setIdOffset(offset);
    m_kernelApiTable->setIdOffset(offset);
    m_copyApiTable->setIdOffset(offset);
    m_opTable->setIdOffset(offset);
    m_apiTable->setIdOffset(offset);

    // Create available datasourced #FIXME: use static initializer
    //m_sources.push_back(new RoctracerDataSource());
    m_sources.push_back(new CuptiDataSource());

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
    }
}

static bool doFinalize = true;
std::mutex finalizeMutex;

void Logger::finalize()
{
    std::lock_guard<std::mutex> guard(finalizeMutex);
    if (doFinalize == true) {
        doFinalize = false;

        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->stopTracing();

        for (auto it = m_sources.begin(); it != m_sources.end(); ++it)
            (*it)->end();

        // Flush recorders
        //const timestamp_t begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t begin_time = clocktime_ns();
        m_stringTable->finalize();
        m_opTable->finalize();		// OpTable before subclassOpTables
        m_kernelApiTable->finalize();
        m_copyApiTable->finalize();
        m_apiTable->finalize();
        //const timestamp_t end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
	//FIXME
        const timestamp_t end_time = clocktime_ns();
        printf("rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);
    }
}

void Logger::createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args)
{
return;
    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = start;
    row.end = end;
    row.apiName_id = m_stringTable->getOrCreate(name);
    row.args_id = m_stringTable->getOrCreate(args);
    row.api_id = 0;

    //printf("overhead: %s (%s) - %f usec\n", name.c_str(), args.c_str(), (end-start) / 1000.0);

    m_apiTable->insertRoctx(row);
}

