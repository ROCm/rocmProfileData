/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <string>
#include <vector>

namespace rpdtracer {

class Slot {
public:
    struct Header {
        int head;
        int tail;
        int capacity;
        size_t elementSize;
        char tag[32];
    };

    Slot(void *memory)
    : m_header(static_cast<Header*>(memory))
    , m_data(static_cast<char*>(memory) + sizeof(Header))
    {}

    int& head() { return m_header->head; }
    int& tail() { return m_header->tail; }
    int capacity() const { return m_header->capacity; }
    size_t elementSize() const { return m_header->elementSize; }
    const char* tag() const { return m_header->tag; }

    template<typename T>
    T* rows() { return static_cast<T*>(m_data); }

private:
    Header *m_header;
    void *m_data;
};


class BufferPool {
public:
    BufferPool();
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    template<typename T>
    Slot* allocate(int count, const char* tag);

private:
    struct SlotEntry {
        void *memory;
        Slot *slot;
        std::function<void()> destructor;
    };
    std::vector<SlotEntry> m_entries;
};


template<typename T>
Slot* BufferPool::allocate(int count, const char* tag)
{
    size_t headerSize = sizeof(Slot::Header);
    size_t dataSize = sizeof(T) * count;
    size_t totalSize = headerSize + dataSize;

    void *memory = std::malloc(totalSize);
    if (!memory)
        throw std::bad_alloc();

    Slot::Header *header = static_cast<Slot::Header*>(memory);
    header->head = 0;
    header->tail = 0;
    header->capacity = count;
    header->elementSize = sizeof(T);
    std::memset(header->tag, 0, sizeof(header->tag));
    std::strncpy(header->tag, tag, sizeof(header->tag) - 1);

    T *rows = reinterpret_cast<T*>(static_cast<char*>(memory) + headerSize);
    for (int i = 0; i < count; ++i)
        new (&rows[i]) T();

    Slot *slot = new Slot(memory);

    m_entries.push_back({memory, slot, [rows, count]() {
        for (int i = 0; i < count; ++i)
            rows[i].~T();
    }});

    return slot;
}

}  // namespace rpdtracer
