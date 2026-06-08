/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "TcpConnection.h"

#include <string>

namespace rpdtracer {

/**************************************************************************
 * NetWriterBackend: serializes rows and sends over TCP to a remote
 * receiver instead of writing to a local SQLite database.
 *
 * One NetWriterBackend per table per process. Each owns a dedicated TCP
 * connection to the receiver on Node 0.
 *
 * Wire protocol
 * -------------
 * After connecting, the client sends a handshake:
 *
 *   [tag: 32 bytes]          Table type identifier (e.g. "OpTable")
 *   [directWrite: 1 byte]    0 = use temp tables, 1 = direct insert
 *
 * Then the client streams batch messages:
 *
 *   [messageSize: 4 bytes]   Total size of this message (including this field)
 *   [idOffset: 8 bytes]      ID offset for foreign key adjustment
 *   [nodeId: 4 bytes]        Node identifier for stride-based adjustments
 *   [rowCount: 4 bytes]      Number of serialized rows in payload
 *   [startIndex: 4 bytes]    Buffer start index (for primary key materialization)
 *   [payload: variable]      rowCount serialized rows (via row::serialize)
 *
 * A flush signal is a batch message with rowCount = 0 and no payload.
 * The receiver cascades the flush to its SqliteWriterBackend.
 *
 * The receiver (Node 0) is the authority on stride values for pid/tid/gpu
 * adjustments. It reads nodeId from each batch and applies:
 *   pid  += nodeId * pidStride
 *   gpuId += nodeId * gpuStride
 *   idOffset += nodeId * nodeIdStride
 **************************************************************************/
class NetWriterBackend : public WriterBackend {
public:
    // tag: table type (e.g. "OpTable"), used in handshake
    // host/port: receiver address
    // directWrite: forwarded to receiver so it creates the right backend
    // serializeRow: function that serializes one row into a ByteBuffer
    using SerializeFn = void (*)(const void *row, ByteBuffer &buf);

    NetWriterBackend(const char *tag, const char *host, int port,
                     bool directWrite, size_t rowSize, SerializeFn serializeRow);
    ~NetWriterBackend() override;

    void writeBatch(void *rows, int start, int end, int capacity) override;
    void flush() override;
    void setIdOffset(sqlite3_int64 offset) override;

private:
    bool ensureConnected();
    void sendHandshake();

    TcpConnection m_conn;
    ByteBuffer m_buf;
    std::string m_host;
    int m_port;
    char m_tag[32];
    bool m_directWrite;
    size_t m_rowSize;
    SerializeFn m_serializeRow;
    sqlite3_int64 m_idOffset{0};
    int m_nodeId{0};
    bool m_handshakeSent{false};
    bool m_connected{false};
};

}  // namespace rpdtracer
