/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <cstdint>
#include <cstring>
#include <sqlite3.h>
#include <string>
#include <vector>

namespace rpdtracer {

class ByteBuffer {
public:
    void writeInt(int v) {
        append(&v, sizeof(v));
    }
    void writeInt64(sqlite3_int64 v) {
        append(&v, sizeof(v));
    }
    void writeBool(bool v) {
        char c = v ? 1 : 0;
        append(&c, 1);
    }
    void writeString(const std::string &s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        append(&len, sizeof(len));
        append(s.data(), len);
    }

    int readInt() {
        int v;
        read(&v, sizeof(v));
        return v;
    }
    sqlite3_int64 readInt64() {
        sqlite3_int64 v;
        read(&v, sizeof(v));
        return v;
    }
    bool readBool() {
        char c;
        read(&c, 1);
        return c != 0;
    }
    std::string readString() {
        uint32_t len;
        read(&len, sizeof(len));
        std::string s(m_data.data() + m_pos, len);
        m_pos += len;
        return s;
    }

    const char* data() const { return m_data.data(); }
    size_t size() const { return m_data.size(); }

    void setData(const char *d, size_t len) {
        m_data.assign(d, d + len);
        m_pos = 0;
    }
    void clear() {
        m_data.clear();
        m_pos = 0;
    }

private:
    void append(const void *p, size_t n) {
        const char *c = static_cast<const char*>(p);
        m_data.insert(m_data.end(), c, c + n);
    }
    void read(void *p, size_t n) {
        std::memcpy(p, m_data.data() + m_pos, n);
        m_pos += n;
    }

    std::vector<char> m_data;
    size_t m_pos{0};
};

}  // namespace rpdtracer
