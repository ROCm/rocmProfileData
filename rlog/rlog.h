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

#include <dlfcn.h>
namespace rlog {

  // API functions -------------------------------------------------------------
  void init();

  void mark(const char *domain, const char *category, const char *apiname, const char *args);
  void mark(const char *category, const char *apiname, const char *args);
  void mark(const char *apiname, const char *args);

  void rangePush(const char *domain, const char *category, const char *apiname, const char *args);
  void rangePush(const char *category, const char *apiname, const char *args);
  void rangePush(const char *apiname, const char *args);

  void rangePop();

  int registerActiveCallback(void (*cb)());
  bool isActive();

  void setDefaultDomain(const char *);
  void setDefaultCategory(const char *);
  const char *getProperty(const char *domain, const char *property, const char *defaultValue);

  int getVersion();
  int localVersion = 1;

  // END API functions ---------------------------------------------------------

namespace {
    const char *domain = "";
    const char *category = "";

    // Static function pointers
    void (*log_mark_) (const char*, const char*, const char*, const char*) = NULL;
    void (*log_rangePush_) (const char*, const char*, const char*, const char*) = NULL;
    void (*log_rangePop_) () = NULL;
    void (*log_registerActiveCallback_) (void (*cb)()) = NULL;
    bool (*log_isActive_) () = NULL;
    const char* (*log_getProperty_) (const char*, const char*, const char*) = NULL;

    // Static function pointers - ROCTX
    void (*roctx_mark_) (const char* message) = NULL;
    void (*roctx_rangePush_) (const char* message) = NULL;
    void (*roctx_rangePop_) () = NULL;
} // namespace



// Load library and look up symbols
void init() {
#if 1
    void (*dl) = dlopen("librlog.so", RTLD_LAZY);
    if (dl) {
        log_mark_ = (void (*)(const char*, const char*, const char*, const char*)) dlsym(dl, "rlog_mark");
        log_rangePush_ = (void (*)(const char*, const char*, const char*, const char*)) dlsym(dl, "rlog_rangePush");
        log_rangePop_ = (void (*)()) dlsym(dl, "rlog_rangePop");
        log_registerActiveCallback_ = (void(*)(void (*cb)())) dlsym(dl, "rlog_registerActiveCallback");
        log_isActive_ = (bool (*)()) dlsym(dl, "rlog_isActive");
        log_getProperty_ = (const char*(*)(const char*, const char*, const char*))  dlsym(dl, "rlog_getProperty");
    }
#endif
#if 0
    void (*dltx) = dlopen("libroctx64.so", RTLD_LAZY);
    if (dltx) {
        roctx_mark_ = (void (*)(const char*)) dlsym(dltx, "roctxMarkA");
        roctx_rangePush_ = (void (*)(const char*)) dlsym(dltx, "roctxRangePushA");
        roctx_rangePop_ = (void (*)()) dlsym(dltx, "roctxRangePop");
    }
#endif
}

void mark(const char *domain, const char *category, const char *apiname, const char *args)
{
    if (log_mark_)
        log_mark_(domain, category, apiname, args);
    if (roctx_mark_) {
        char buff[4096];
        snprintf(buff, 4096, "%s : %s : api = %s | %s", domain, category, apiname, args);
        roctx_mark_(buff); 
    }
}

void mark(const char *category, const char *apiname, const char *args)
{
    mark(domain, category, apiname, args);
}

void mark(const char *apiname, const char *args)
{
    mark(domain, category, apiname, args);
}

void rangePush(const char *domain, const char *category, const char *apiname, const char *args)
{
//fprintf(stderr, "rangePush %p\n", log_rangePush_);
    if (log_rangePush_)
       log_rangePush_(domain, category, apiname, args);
    if (roctx_rangePush_) {
        char buff[4096];
        snprintf(buff, 4096, "%s : %s : api = %s | %s", domain, category, apiname, args);
        roctx_rangePush_(buff);
    }
}

void rangePush(const char *category, const char *apiname, const char *args)
{
    rangePush(domain, category, apiname, args);
}

void rangePush(const char *apiname, const char *args)
{
    rangePush(domain, category, apiname, args);
}

void rangePop()
{
    if (log_rangePop_)
        log_rangePop_();
    if (roctx_rangePop_)
        roctx_rangePop_();
}


int registerActiveCallback(void (*cb)())
{
    if (log_registerActiveCallback_) {
        log_registerActiveCallback_(cb);
        return 0;
    }
    else
        return -1;
}

bool isActive()
{
    if (log_isActive_)
        return log_isActive_();
    else
        return false;
}

void setDefaultDomain(const char* ddomain)
{
    domain = ddomain;
}

void setDefaultCategory(const char* dcat)
{
    category = dcat;
}

// FIXME: lifetime
const char *getProperty(const char *domain, const char *property, const char *defaultValue)
{
    if (log_getProperty_)
        return log_getProperty_(domain, property, defaultValue);
    return defaultValue;
}

} // namespace rlog
