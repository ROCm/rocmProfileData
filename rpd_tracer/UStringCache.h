/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sqlite3.h>
#include "Table.h"

namespace rpdtracer {

struct UStringCache
{
    struct Entry {
        std::string content;
        sqlite3_int64 id;
        bool valid;
    };
    static const int CAPACITY = 4096;
    Entry entries[CAPACITY] = {};
    uint64_t generation = UINT64_MAX;
    uint64_t hits = 0;
    uint64_t misses = 0;
    ~UStringCache() { if (hits + misses > 0) fprintf(stderr, "UStringCache: %lu hits, %lu misses (%.1f%% hit rate)\n", hits, misses, 100.0 * hits / (hits + misses)); }

    sqlite3_int64 lookup(const std::string &str, UStringTable &table, uint64_t currentGen)
    {
        checkGeneration(currentGen);
        uint64_t h = hash(str.data(), str.size());
        int slot = h % CAPACITY;
        Entry &e = entries[slot];
        if (e.valid && e.content == str) {
            ++hits;
            return e.id;
        }
        ++misses;
        sqlite3_int64 id = table.create(str);
        e.content = str;
        e.id = id;
        e.valid = true;
        return id;
    }

    sqlite3_int64 lookup(const char *str, UStringTable &table, uint64_t currentGen)
    {
        if (str[0] == '\0')
            return EMPTY_STRING_ID;
        checkGeneration(currentGen);
        uint64_t h = hash(str, strlen(str));
        int slot = h % CAPACITY;
        Entry &e = entries[slot];
        if (e.valid && e.content == str) {
            ++hits;
            return e.id;
        }
        ++misses;
        sqlite3_int64 id = table.create(std::string(str));
        e.content = str;
        e.id = id;
        e.valid = true;
        return id;
    }

private:
    void checkGeneration(uint64_t currentGen)
    {
        if (generation != currentGen) {
            for (int i = 0; i < CAPACITY; ++i)
                entries[i].valid = false;
            generation = currentGen;
            hits = 0;
            misses = 0;
        }
    }

    static uint64_t hash(const char *data, size_t len)
    {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= static_cast<uint64_t>(data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

}    // namespace rpdtracer
