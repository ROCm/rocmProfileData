// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

struct sqlite3;

extern "C" {
    void rpdstart();
    void rpdstop();
    void rpdflush();
    void rpd_setConfig(const char *property, const char *value);
    sqlite3 *rpd_getConnection();
    void rpd_resetStorage();
}

namespace rpdtracer {

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);

}    // namespace rpdtracer
