/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <string>
#include <sqlite3.h>
#include "Table.h"

namespace rpdtracer {

struct LocalStringCache
{
    struct Entry {
        const char *ptr;
        std::string content;
        sqlite3_int64 id;
        uint64_t generation;
    };
    static const int CAPACITY = 128;
    Entry entries[CAPACITY];
    int count = 0;
    int writeIdx = 0;

    sqlite3_int64 lookup(const std::string &str, StringTable &table, uint64_t currentGen)
    {
        return lookup(str.c_str(), table, currentGen);
    }

    sqlite3_int64 lookup(const char *str, StringTable &table, uint64_t currentGen)
    {
        for (int i = 0; i < count; ++i) {
            if (entries[i].ptr == str) {
                if (entries[i].generation != currentGen) {
                    entries[i].id = table.getOrCreate(entries[i].content);
                    entries[i].generation = currentGen;
                }
                return entries[i].id;
            }
        }
        for (int i = 0; i < count; ++i) {
            if (entries[i].content == str) {
                entries[i].ptr = str;
                if (entries[i].generation != currentGen) {
                    entries[i].id = table.getOrCreate(entries[i].content);
                    entries[i].generation = currentGen;
                }
                return entries[i].id;
            }
        }
        sqlite3_int64 id = table.getOrCreate(std::string(str));
        if (count < CAPACITY) {
            entries[count++] = {str, std::string(str), id, currentGen};
        } else {
            entries[writeIdx] = {str, std::string(str), id, currentGen};
            writeIdx = (writeIdx + 1) % CAPACITY;
        }
        return id;
    }

    void reset() { count = 0; writeIdx = 0; }
};

}    // namespace rpdtracer
