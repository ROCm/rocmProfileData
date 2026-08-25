/*********************************************************************************
* Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "RlogDataSource.h"

#include "Logger.h"
#include "Utility.h"

#include <cstdio>
#include <fmt/format.h>

#include "../rlog/RLogger.h"	// FIXME

// Create a factory for the Logger to locate and use
extern "C" {
    DataSource *RlogDataSourceFactory() { return new RlogDataSource(); }
}  // extern "C"


void RlogDataSource::init()
{
}

void RlogDataSource::end()
{
}

void RlogDataSource::startTracing()
{
    rlog::RLogger::singleton().addLogger(*this);
}

void RlogDataSource::stopTracing()
{
    rlog::RLogger::singleton().removeLogger(*this);
}

void RlogDataSource::flush()
{
}


void RlogDataSource::mark(const char *domain, const char *category, const char *apiname, const char *args)
{
    ::Logger &logger = ::Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
    row.apiName_id = markerId;
    row.args_id = EMPTY_STRING_ID;
    row.api_id = 0;

    auto message = fmt::format("{}::{}::{} | {}", domain, category, apiname, args);

    row.args_id = logger.stringTable().getOrCreate(message);
    logger.apiTable().insertRoctx(row);
}

void RlogDataSource::rangePush(const char *domain, const char *category, const char *apiname, const char *args)
{
    ::Logger &logger = ::Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
    row.apiName_id = markerId;
    row.args_id = EMPTY_STRING_ID;
    row.api_id = 0;

    auto message = fmt::format("{}::{}::{} | {}", domain, category, apiname, args);

    row.args_id = logger.stringTable().getOrCreate(message);
    logger.apiTable().pushRoctx(row);
}

void RlogDataSource::rangePop()
{
    ::Logger &logger = ::Logger::singleton();

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = clocktime_ns();
    row.end = row.start;
    static sqlite3_int64 markerId = logger.stringTable().getOrCreate(std::string("UserMarker"));
    row.apiName_id = markerId;
    row.args_id = EMPTY_STRING_ID;
    row.api_id = 0;

    logger.apiTable().popRoctx(row);
}


