/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <atomic>
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


struct PoolHeader {
    std::atomic<int> ready{0};
    std::atomic<int> slotCount{0};
    std::atomic<size_t> allocOffset{0};
    size_t totalSize{0};
};


class BufferPool {
public:
    BufferPool();
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    void initShared(const char *dbFilename);

    template<typename T>
    Slot* allocate(int count, const char* tag);

private:
    struct SlotEntry {
        void *memory;
        Slot *slot;
        std::function<void()> destructor;
    };
    std::vector<SlotEntry> m_entries;

    bool m_shared{false};
    void *m_shmBase{nullptr};
    int m_shmFd{-1};
    std::string m_shmName;
    bool m_creator{false};
};


template<typename T>
Slot* BufferPool::allocate(int count, const char* tag)
{
    size_t headerSize = sizeof(Slot::Header);
    size_t dataSize = sizeof(T) * count;
    size_t slotSize = headerSize + dataSize;

    void *memory;

    if (!m_shared) {
        memory = std::malloc(slotSize);
        if (!memory)
            throw std::bad_alloc();
    } else {
        PoolHeader *pool = static_cast<PoolHeader*>(m_shmBase);
        size_t offset = pool->allocOffset.fetch_add(slotSize, std::memory_order_relaxed);
        if (offset + slotSize > pool->totalSize)
            throw std::bad_alloc();
        memory = static_cast<char*>(m_shmBase) + offset;
    }

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

    if (m_shared) {
        PoolHeader *pool = static_cast<PoolHeader*>(m_shmBase);
        pool->slotCount.fetch_add(1, std::memory_order_release);
    }

    return slot;
}

}  // namespace rpdtracer
