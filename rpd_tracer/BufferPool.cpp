/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "BufferPool.h"

#include <cstdlib>

namespace rpdtracer {

BufferPool::BufferPool() = default;

BufferPool::~BufferPool()
{
    for (auto it = m_entries.rbegin(); it != m_entries.rend(); ++it) {
        it->destructor();
        std::free(it->memory);
        delete it->slot;
    }
    m_entries.clear();
}

}  // namespace rpdtracer
