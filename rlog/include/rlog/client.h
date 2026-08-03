// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rlog {

  // API functions -------------------------------------------------------------
  // Must be called once before any other rlog functions; not thread-safe.
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

  // Must be called before any concurrent logging begins; not thread-safe.
  void setDefaultDomain(const char *);
  void setDefaultCategory(const char *);
  const char *getProperty(const char *domain, const char *property, const char *defaultValue);

  //int getVersion();	// FIXME
  //int localVersion();

  enum Api {
      Rlog,
      Roctx,
      Nvtx,
      API_COUNT
  };
  bool enabled(Api);
  void setEnabled(Api, bool enable);

  // END API functions ---------------------------------------------------------

} // namespace rlog

/*

ENV variables

*
* Choose a non-default location for a legacy api library
*   Value should be an absolute path
*
RLOG_NVTX_LIBPATH
RLOG_ROCTX_LIBPATH

*
* Force a client application to log to a legacy api. 
*   Useful when using profilers/tools that are not rlog aware.
*   Logging state is jammed on.
*   Value should be non-zero to force on
*
RLOG_FORCE_ROCTX
RLOG_FORCE_NVTX

*/
