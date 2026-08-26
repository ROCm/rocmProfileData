/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <sqlite3.h>

namespace rpdtracer {

class WriterBackend {
public:
    virtual void writeBatch(void *rows, int start, int end, int capacity) = 0;
    virtual void flush() = 0;
    virtual void setIdOffset(sqlite3_int64 offset) = 0;
    virtual ~WriterBackend() = default;
};

}  // namespace rpdtracer
