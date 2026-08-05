// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

extern "C" {
    void rpdstart();
    void rpdstop();
    void rpdflush();
    void rpd_mark(const char *domain, const char *apiName, const char* args);
    void rpd_rangePush(const char *domain, const char *apiName, const char* args);
    void rpd_rangePop();
}

namespace rpdtracer {

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);

}    // namespace rpdtracer
