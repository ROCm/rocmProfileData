/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "NetWriterBackend.h"
#include "Utility.h"

#include <cstring>

namespace rpdtracer {

NetWriterBackend::NetWriterBackend(const char *tag, const char *host, int port,
                                   bool directWrite, size_t rowSize,
                                   SerializeFn serializeRow)
: m_host(host)
, m_port(port)
, m_directWrite(directWrite)
, m_rowSize(rowSize)
, m_serializeRow(serializeRow)
{
    std::memset(m_tag, 0, sizeof(m_tag));
    std::strncpy(m_tag, tag, sizeof(m_tag) - 1);

    const char *nodeIdStr = getenv("RPDT_NODE_ID");
    if (nodeIdStr)
        m_nodeId = atoi(nodeIdStr);
}

NetWriterBackend::~NetWriterBackend()
{
    m_conn.close();
}

void NetWriterBackend::setIdOffset(sqlite3_int64 offset)
{
    m_idOffset = offset;
}

bool NetWriterBackend::ensureConnected()
{
    if (m_connected)
        return true;

    // Retry with backoff — receiver may not be listening yet
    for (int attempt = 0; attempt < 10; ++attempt) {
        if (m_conn.connect(m_host.c_str(), m_port)) {
            m_connected = true;
            sendHandshake();
            return true;
        }
        usleep(100000 * (attempt + 1));  // 100ms, 200ms, 300ms...
    }

    rpdLog("NetWriterBackend: failed to connect to %s:%d after retries\n", m_host.c_str(), m_port);
    return false;
}

void NetWriterBackend::sendHandshake()
{
    /*  Handshake: [tag: 32 bytes][directWrite: 1 byte]  */
    m_conn.send(m_tag, 32);
    char dw = m_directWrite ? 1 : 0;
    m_conn.send(&dw, 1);
    m_handshakeSent = true;
}

void NetWriterBackend::writeBatch(void *rowData, int start, int end, int capacity)
{
    if (!ensureConnected())
        return;

    int rowCount = end - start + 1;
    const char *base = static_cast<const char*>(rowData);

    // Serialize all rows into the buffer
    m_buf.clear();
    for (int i = start; i <= end; ++i) {
        const void *row = base + (i % capacity) * m_rowSize;
        m_serializeRow(row, m_buf);
    }

    /*  Batch message:
     *    [messageSize: 4]  total message bytes (header + payload)
     *    [idOffset: 8]     ID offset for this batch
     *    [nodeId: 4]       node identifier for stride adjustments
     *    [rowCount: 4]     number of rows
     *    [payload]         serialized row data
     */
    uint32_t payloadSize = static_cast<uint32_t>(m_buf.size());
    uint32_t messageSize = 4 + 8 + 4 + 4 + payloadSize;

    m_conn.send(&messageSize, 4);
    m_conn.send(&m_idOffset, 8);
    m_conn.send(&m_nodeId, 4);
    m_conn.send(&rowCount, 4);
    m_conn.send(m_buf.data(), payloadSize);
}

void NetWriterBackend::flush()
{
    if (!m_connected)
        return;

    /*  Flush signal: batch message with rowCount = 0, no payload  */
    uint32_t messageSize = 4 + 8 + 4 + 4;
    uint32_t rowCount = 0;

    m_conn.send(&messageSize, 4);
    m_conn.send(&m_idOffset, 8);
    m_conn.send(&m_nodeId, 4);
    m_conn.send(&rowCount, 4);
}

}  // namespace rpdtracer
