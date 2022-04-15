/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once


extern "C" {
    void rpdstart();
    void rpdstop();
}

void createOverheadRecord(uint64_t start, uint64_t end, const std::string &name, const std::string &args);
