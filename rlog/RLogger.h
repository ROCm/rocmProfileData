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
#pragma once

#include "Logger.h"

namespace rlog {

class RLoggerPrivate;
class RLogger: public Logger
{
public:
    static RLogger& singleton();

    // External marker api
    virtual void mark(const char *domain, const char *category, const char *apiName, const char* args) override;
    virtual void rangePush(const char *domain, const char *category, const char *apiName, const char* args) override;
    virtual void rangePop() override;

    // Add or remove a logger - ref counted per logger
    void addLogger(Logger &logger);
    void removeLogger(Logger &logger);

    // Active is true when any logger in present
    void registerActiveCallback(void (*cb)());
    bool isActive();

    // Properties
    const char *getProperty(const char *domain, const char *property, const char *defaultValue);

private:
    RLogger();
    virtual ~RLogger();

    RLoggerPrivate *d;
    friend class RLoggerPrivate;

    static void rlogInit() __attribute__((constructor));
    static void rlogFinalize() __attribute__((destructor));

    void init();
    void finalize();
};

}  // namespace rlog
