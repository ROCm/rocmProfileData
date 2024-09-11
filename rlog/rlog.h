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

#include <dlfcn.h>

// Static function pointer
void (*log_mark_) (const char*, const char*, const char*) = NULL;
void (*log_rangePush_) (const char*, const char*, const char*) = NULL;
void (*log_rangePop_) () = NULL;
void (*registerActiveCallback_) (void (*cb)()) = NULL;

// Load library and look up symbols
void init_rlog() {
    void (*dl) = dlopen("librlog.so", RTLD_LAZY);
    if (dl) {
        log_mark_ = (void (*)(const char*, const char*, const char*)) dlsym(dl, "log_mark");
        log_rangePush_ = (void (*)(const char*, const char*, const char*)) dlsym(dl, "log_rangePush");
        log_rangePop_ = (void (*)()) dlsym(dl, "log_rangePop");
    }
}

// Application usable functions
void log_mark(const char *domain, const char *apiname, const char *args) {
    if (log_mark_)
        log_mark_(domain, apiname, args);
}
void log_rangePush(const char *domain, const char *apiname, const char *args) {
    if (log_rangePush_)
        log_rangePush_(domain, apiname, args);
}
void log_rangePop() {
    if (log_rangePop_)
       log_rangePop_();
}

int registerActiveCallback(void (*cb)()) {
    if (registerActiveCallback_) {
        registerActiveCallback_(cb);
        return 0;
    }
    else
        return -1;
}


