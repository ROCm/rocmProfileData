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

#include <cstdio>

#include "RLogger.h"

extern "C" {
void log_mark(const char *domain, const char *apiName, const char* args)
{
    fprintf(stderr, "MARK\n");
}
void log_rangePush(const char *domain, const char *apiName, const char* args)
{
    fprintf(stderr, "PUSH\n");
}
void log_rangePop()
{
    fprintf(stderr, "POP\n");
}
}


RLogger& RLogger::singleton()
{
    static RLogger logger;
    return logger;
}


void RLogger::rlogInit() {
    RLogger::singleton().init();
}

void RLogger::rlogFinalize() {
    RLogger::singleton().finalize();
}

void RLogger::init()
{
    fprintf(stderr, "RLogger::init\n");
}

void RLogger::finalize()
{
    fprintf(stderr, "RLogger::finalize\n");
}
