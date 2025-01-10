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

#include <sqlite3.h>
#include "Logger.h"

#ifdef RPD_STACKFRAME_SUPPORT
#include <cpptrace/cpptrace.hpp>
#include <sstream>
#include <iostream>

// FIXME: can we avoid shutdown corruption?
// Other rocm libraries crashing on unload
// libsqlite unloading before we are done using it
// Current workaround: register an onexit function when first activity is delivered back
//                     this let's us unload first, or close to.
// New workaround: register 4 times, only finalize once.  see register_once

static std::once_flag registerDoubleAgain_once;

int unwind(Logger &logger, const char *api, const sqlite_int64 api_id) {

    std::cout << "unwinding" << std::endl;
    if (!logger.writeStackFrames()) return 0;

#if 0
    // for reference: full stack w/o manipulations
    const std::string stack1 = cpptrace::generate_trace(0).to_string(false);
    std::cout << stack1 << std::endl;
    if (true) return 0;
#endif

    // strip out the top frames that only point into roctracer/rpd, do not add color
    const std::string stack = cpptrace::generate_trace(3).to_string(false);
    /*
     * returns:
     * Stack trace (most recent call first):
     * #0 0x00007f3a1d2f4447 at /opt/rocm/lib/libamdhip64.so.6
     * #1 0x00000000002055cf in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:170:5
     * #2 0x00007f3a1cb561c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
     * #3 0x00007f3a1cb5628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
     * #4 0x0000000000204b04 at /root/rocm-examples/Applications/bitonic_sort/applications_bitonic_sort
     *
     * need to get rid of the first line
     * should inject api into #0 frame as "in $api at"
     */
    std::istringstream iss(stack);
    std::string line;
    std::getline(iss, line); // get rid of "Stack trace (most recent call first):"

    std::getline(iss, line);
    std::string s1 = line.substr(0,21);
    std::string s2 = line.substr(21);

    std::string fixed = s1 + " in " + api + "()" + s2;

    StackFrameTable::row frame0;
    frame0.api_id = api_id;
    frame0.depth = 0;
    frame0.name_id = logger.stringTable().getOrCreate(fixed.c_str());
    logger.stackFrameTable().insert(frame0);

    int n = 1;
    while ( std::getline(iss, line) ) {
        if (line.empty())
            continue;
        StackFrameTable::row frame;
        frame.api_id = api_id;
        frame.depth = n;
        frame.name_id = logger.stringTable().getOrCreate(line.c_str());
        logger.stackFrameTable().insert(frame);

        n++;
    }

    std::call_once(registerDoubleAgain_once, atexit, Logger::rpdFinalize);

    std::cout << "returning" << std::endl;

    return 0;
}

#else

int unwind(Logger &logger, const char *api, const sqlite_int64 api_id) {
    std::cout << "not supported" << std:endl;
    	return 0;
}

#endif

