/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "RemoteDataSource.h"
#include "Table.h"
#include "Utility.h"

#include <fmt/format.h>
#include <cstring>
#include <vector>
#include <poll.h>
#include <unistd.h>

namespace rpdtracer {

/**************************************************************************
 * Per-table deserialize-and-write functions.
 *
 * Each reads rowCount rows from the ByteBuffer, deserializes into a
 * temporary array, sets idOffset on the backend, and calls writeBatch.
 **************************************************************************/

/*  Stride values for multi-node ID uniqueness.
 *  Read from rocpd_metadata at init time (set during schema creation).
 *  Node 0 is the authority — all adjustments happen here.
 *
 *  pid_stride (default 10000000): pid += nodeId * pid_stride
 *  gpu_stride (default 1000):     gpuId += nodeId * gpu_stride
 *
 *  The napi/nop SQL views decode these via integer division:
 *    node = pid / pid_stride,  pid = pid % pid_stride
 *    node = gpuId / gpu_stride, gpuId = gpuId % gpu_stride
 */
static int s_pidStride = 10000000;
static int s_gpuStride = 1000;

template<typename RowType>
static void deserializeAndWrite(ByteBuffer &buf, int rowCount,
                                sqlite3_int64 idOffset, int nodeId,
                                int startIndex, WriterBackend *backend)
{
    std::vector<RowType> rows(rowCount);
    for (int i = 0; i < rowCount; ++i)
        rows[i].deserialize(buf);
    backend->setIdOffset(idOffset + startIndex);
    backend->writeBatch(rows.data(), 0, rowCount - 1, rowCount);
}

static void deserializeStringTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<StringTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}
static void deserializeUStringTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<UStringTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}
static void deserializeApiTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    // ApiTable has pid and tid that need node-based stride adjustment
    std::vector<ApiTable::row> rows(rowCount);
    for (int i = 0; i < rowCount; ++i) {
        rows[i].deserialize(buf);
        rows[i].pid += nodeId * s_pidStride;
        rows[i].tid += nodeId * s_pidStride;
    }
    backend->setIdOffset(idOffset + startIndex);
    backend->writeBatch(rows.data(), 0, rowCount - 1, rowCount);
}
static void deserializeKernelApiTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<KernelApiTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}
static void deserializeCopyApiTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<CopyApiTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}
static void deserializeOpTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    // OpTable has gpuId that needs node-based stride adjustment
    std::vector<OpTable::row> rows(rowCount);
    for (int i = 0; i < rowCount; ++i) {
        rows[i].deserialize(buf);
        rows[i].gpuId += nodeId * s_gpuStride;
    }
    backend->setIdOffset(idOffset + startIndex);
    backend->writeBatch(rows.data(), 0, rowCount - 1, rowCount);
}
static void deserializeMonitorTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<MonitorTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}
static void deserializeStackFrameTable(ByteBuffer &buf, int rowCount,
    sqlite3_int64 idOffset, int nodeId, int startIndex, WriterBackend *backend)
{
    deserializeAndWrite<StackFrameTable::row>(buf, rowCount, idOffset, nodeId, startIndex, backend);
}


/**************************************************************************
 * RemoteDataSource implementation
 **************************************************************************/

RemoteDataSource::RemoteDataSource() = default;

RemoteDataSource::~RemoteDataSource()
{
    end();
    for (auto *c : m_recvConns)
        delete c;
    m_recvConns.clear();
    for (auto &pair : m_channels) {
        if (pair.second->backend)
            delete pair.second->backend;
        delete pair.second;
    }
    m_channels.clear();
}

void RemoteDataSource::registerChannel(const char *tag, bool directWrite,
                                       DeserializeAndWriteFn fn)
{
    auto *ch = new WriterChannel();
    ch->deserializeAndWrite = fn;

    // Each channel gets its own SqliteWriterBackend (own connection + temp tables).
    // This is the key to thread-local INSERT VALUES with no contention.
    if (std::string(tag) == "StringTable")
        ch->backend = StringTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "UStringTable")
        ch->backend = UStringTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "ApiTable")
        ch->backend = ApiTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "KernelApiTable")
        ch->backend = KernelApiTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "CopyApiTable")
        ch->backend = CopyApiTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "OpTable")
        ch->backend = OpTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "MonitorTable")
        ch->backend = MonitorTable::createWriterBackend(m_basefile.c_str(), directWrite);
    else if (std::string(tag) == "StackFrameTable")
        ch->backend = StackFrameTable::createWriterBackend(m_basefile.c_str(), directWrite);

    m_channels[tag] = ch;
}

void RemoteDataSource::init()
{
    const char *portStr = getenv("RPDT_LOGAGG_PORT");
    if (!portStr) {
        fprintf(stderr, "[%d] RemoteDataSource: RPDT_LOGAGG_PORT not set, skipping\n", getpid());
        return;
    }
    m_port = atoi(portStr);

    m_basefile = getConfig("RPDT_FILENAME", "filename", "./trace.rpd");
    m_directWrite = (atoi(getConfig("RPDT_DIRECTWRITE", "directwrite", "0")) != 0);

    m_resource = new DbResource(m_basefile, std::string("remote_active"));
    if (!m_resource->tryLock()) {
        delete m_resource;
        m_resource = nullptr;
        return;
    }

    {
        sqlite3 *mdb = nullptr;
        if (rpdSqliteOpen(m_basefile.c_str(), &mdb) == SQLITE_OK) {
            std::string sql = fmt::format("INSERT INTO rocpd_metadata(tag, value) VALUES ('remote_delegate', 'pid={}')", getpid());
            sqlite3_exec(mdb, sql.c_str(), nullptr, nullptr, nullptr);
            sqlite3_close(mdb);
        }
    }

    // Read stride values from metadata table (set during schema creation)
    sqlite3 *db;
    if (rpdSqliteOpen(m_basefile.c_str(), &db) == SQLITE_OK) {
        sqlite3_stmt *stmt;
        if (sqlite3_prepare_v2(db, "SELECT value FROM rocpd_metadata WHERE tag=?", -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, "pid_stride", -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW)
                s_pidStride = atoi(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
            sqlite3_reset(stmt);
            sqlite3_bind_text(stmt, 1, "gpu_stride", -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW)
                s_gpuStride = atoi(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
        rpdLog("RemoteDataSource: pid_stride=%d gpu_stride=%d\n", s_pidStride, s_gpuStride);
    }

    // Try to bind — only the delegate succeeds. Other processes bail here.
    if (!m_listener.listen(m_port)) {
        fprintf(stderr, "[%d] RemoteDataSource: port %d in use (delegate already running)\n", getpid(), m_port);
        return;
    }

    // Register a writer channel for each table type
    registerChannel("StringTable", m_directWrite, deserializeStringTable);
    registerChannel("UStringTable", m_directWrite, deserializeUStringTable);
    registerChannel("ApiTable", m_directWrite, deserializeApiTable);
    registerChannel("KernelApiTable", m_directWrite, deserializeKernelApiTable);
    registerChannel("CopyApiTable", m_directWrite, deserializeCopyApiTable);
    registerChannel("OpTable", m_directWrite, deserializeOpTable);
    registerChannel("MonitorTable", m_directWrite, deserializeMonitorTable);
    registerChannel("StackFrameTable", m_directWrite, deserializeStackFrameTable);

    // Start writer threads
    for (auto &pair : m_channels) {
        pair.second->worker = new std::thread(&RemoteDataSource::writerLoop, pair.second);
    }

    m_running = true;
    m_acceptThread = new std::thread(&RemoteDataSource::acceptLoop, this);
    fprintf(stderr, "[%d] RemoteDataSource: listening on port %d\n", getpid(), m_port);
}

void RemoteDataSource::end()
{
    if (!m_running)
        return;
    fprintf(stderr, "[%d] RemoteDataSource::end() starting\n", getpid());
    m_running = false;

    fprintf(stderr, "[%d] end: closing listener\n", getpid());
    m_listener.close();
    if (m_acceptThread) {
        if (m_acceptThread->joinable())
            m_acceptThread->join();
        delete m_acceptThread;
        m_acceptThread = nullptr;
    }

    fprintf(stderr, "[%d] end: accept thread joined, closing recv connections\n", getpid());
    // Close all recv connections to unblock recv threads
    {
        std::lock_guard<std::mutex> lock(m_connMutex);
        for (auto *c : m_recvConns)
            c->close();
    }

    fprintf(stderr, "[%d] end: joining %zu recv threads\n", getpid(), m_recvThreads.size());
    // Join recv threads (now unblocked by connection close)
    for (auto *t : m_recvThreads) {
        if (t->joinable())
            t->join();
        delete t;
    }
    m_recvThreads.clear();
    m_recvConns.clear();

    fprintf(stderr, "[%d] end: recv threads done, signaling writers\n", getpid());
    // Signal writer threads to drain and exit
    for (auto &pair : m_channels) {
        WriterChannel *ch = pair.second;
        if (ch->worker) {
            {
                std::lock_guard<std::mutex> lock(ch->mutex);
                ch->done = true;
            }
            ch->cv.notify_one();
            fprintf(stderr, "[%d] end: signaled '%s'\n", getpid(), pair.first.c_str());
        }
    }
    for (auto &pair : m_channels) {
        if (pair.second->worker) {
            fprintf(stderr, "[%d] end: joining '%s'\n", getpid(), pair.first.c_str());
            if (pair.second->worker->joinable())
                pair.second->worker->join();
            delete pair.second->worker;
            pair.second->worker = nullptr;
            fprintf(stderr, "[%d] end: joined '%s'\n", getpid(), pair.first.c_str());
        }
    }
    fprintf(stderr, "[%d] end: all writers done\n", getpid());

    if (m_resource != nullptr) {
        m_resource->unlock();
        delete m_resource;
        m_resource = nullptr;
    }
}

void RemoteDataSource::flush()
{
    for (auto &pair : m_channels)
        pair.second->backend->flush();
}

/**************************************************************************
 * Accept loop: runs in its own thread. Accepts TCP connections, reads
 * the handshake to determine the table type, then spawns a recv thread.
 *
 * Wire protocol — handshake (sent by NetWriterBackend on connect):
 *   [tag: 32 bytes]          Table type identifier
 *   [directWrite: 1 byte]    0 = temp tables, 1 = direct insert
 **************************************************************************/
void RemoteDataSource::acceptLoop()
{
    while (m_running) {
        // Poll with timeout so we can check m_running periodically
        struct pollfd pfd;
        pfd.fd = m_listener.fd();
        pfd.events = POLLIN;
        int ret = poll(&pfd, 1, 500);  // 500ms timeout
        if (ret <= 0)
            continue;

        TcpConnection *conn = m_listener.accept();
        if (!conn)
            break;

        // Read handshake
        char tag[32];
        char directWrite;
        if (!conn->recv(tag, 32) || !conn->recv(&directWrite, 1)) {
            rpdLog("RemoteDataSource: handshake failed\n");
            delete conn;
            continue;
        }

        auto it = m_channels.find(tag);
        if (it == m_channels.end()) {
            rpdLog("RemoteDataSource: unknown table type '%s'\n", tag);
            delete conn;
            continue;
        }

        fprintf(stderr, "[%d] RemoteDataSource: accepted connection for '%s'\n", getpid(), tag);

        // Track connection and spawn a recv thread
        {
            std::lock_guard<std::mutex> lock(m_connMutex);
            m_recvConns.push_back(conn);
        }
        auto *t = new std::thread(&RemoteDataSource::recvLoop, this, conn, it->second);
        m_recvThreads.push_back(t);
    }
}

/**************************************************************************
 * Recv loop: reads batch messages from one connection and enqueues them
 * to the writer channel.
 *
 * Wire protocol — batch messages (after handshake):
 *   [messageSize: 4 bytes]   Total message size (header + payload)
 *   [idOffset: 8 bytes]      ID offset for foreign key adjustment
 *   [rowCount: 4 bytes]      Number of rows (0 = flush signal)
 *   [payload: variable]      Serialized row data
 **************************************************************************/
void RemoteDataSource::recvLoop(TcpConnection *conn, WriterChannel *channel)
{
    while (m_running) {
        // Read message header
        uint32_t messageSize;
        if (!conn->recv(&messageSize, 4))
            break;

        sqlite3_int64 idOffset;
        if (!conn->recv(&idOffset, 8))
            break;

        int nodeId;
        if (!conn->recv(&nodeId, 4))
            break;

        int rowCount;
        if (!conn->recv(&rowCount, 4))
            break;

        int startIndex;
        if (!conn->recv(&startIndex, 4))
            break;

        if (rowCount == 0) {
            // Flush signal: cascade to the writer's backend
            channel->backend->flush();
            continue;
        }

        // Read payload
        uint32_t payloadSize = messageSize - 4 - 8 - 4 - 4 - 4;
        BatchItem item;
        item.idOffset = idOffset;
        item.nodeId = nodeId;
        item.rowCount = rowCount;
        item.startIndex = startIndex;

        // Read raw bytes into the BatchItem's buffer
        std::vector<char> payload(payloadSize);
        if (!conn->recv(payload.data(), payloadSize))
            break;
        item.data.setData(payload.data(), payloadSize);

        // Enqueue to writer channel
        {
            std::lock_guard<std::mutex> lock(channel->mutex);
            channel->queue.push(std::move(item));
        }
        channel->cv.notify_one();
    }

}

/**************************************************************************
 * Writer loop: drains the batch queue for one table type, deserializes
 * rows, and writes to SQLite via the channel's WriterBackend.
 *
 * INSERT VALUES goes into thread-local temp tables (no contention).
 * Periodic flush (INSERT INTO SELECT) is the only contention point.
 **************************************************************************/
void RemoteDataSource::writerLoop(WriterChannel *channel)
{
    while (true) {
        BatchItem item;
        {
            std::unique_lock<std::mutex> lock(channel->mutex);
            channel->cv.wait(lock, [channel]() {
                return !channel->queue.empty() || channel->done;
            });
            if (channel->queue.empty() && channel->done)
                break;
            item = std::move(channel->queue.front());
            channel->queue.pop();
        }

        const timestamp_t begin = clocktime_ns();
        channel->deserializeAndWrite(item.data, item.rowCount,
                                     item.idOffset, item.nodeId, item.startIndex, channel->backend);
        const timestamp_t end = clocktime_ns();
        char buff[256];
        std::snprintf(buff, sizeof(buff), "node=%d count=%d", item.nodeId, item.rowCount);
        createOverheadRecord(begin, end, "RemoteDataSource::write", buff);
    }

    // Drain remaining items
    std::lock_guard<std::mutex> lock(channel->mutex);
    while (!channel->queue.empty()) {
        BatchItem &item = channel->queue.front();
        channel->deserializeAndWrite(item.data, item.rowCount,
                                     item.idOffset, item.nodeId, item.startIndex, channel->backend);
        channel->queue.pop();
    }

    // Final flush: temp tables → main tables
    channel->backend->flush();
}


// Factory function for dynamic loading via dlsym
extern "C" {
DataSource* RemoteDataSourceFactory() {
    return new RemoteDataSource();
}
}

}  // namespace rpdtracer
