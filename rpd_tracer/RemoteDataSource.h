/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * RemoteDataSource: TCP receiver that accepts serialized profiling data
 * from remote nodes and writes it to the local SQLite database.
 *
 * Runs as a single delegate on Node 0 (elected via DbResource). Remote
 * nodes connect with one TCP connection per table type, sending batched
 * serialized rows via the NetWriterBackend protocol.
 *
 * Architecture:
 *   - Accept thread: listens for connections, reads handshake, spawns
 *     a recv thread per connection.
 *   - Recv threads: read batch messages, enqueue raw bytes to the
 *     matching WriterChannel.
 *   - Writer threads (8, one per table type): dequeue batches,
 *     deserialize rows, write to SQLite via SqliteWriterBackend.
 *     Each has its own connection + temp tables, so INSERT VALUES
 *     is thread-local with zero contention. Only the periodic
 *     INSERT INTO ... SELECT flush touches the main tables.
 *
 * If the delegate crashes, data in-flight is lost. This is acceptable
 * for a profiling tool.
 **************************************************************************/
#pragma once

#include "DataSource.h"
#include "DbResource.h"
#include "WriterBackend.h"
#include "ByteBuffer.h"
#include "TcpConnection.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rpdtracer {

class RemoteDataSource : public DataSource
{
public:
    RemoteDataSource();
    ~RemoteDataSource();

    void init() override;
    void end() override;
    void startTracing() override {}
    void stopTracing() override {}
    void flush() override;

private:
    struct BatchItem {
        ByteBuffer data;
        sqlite3_int64 idOffset;
        int nodeId;
        int rowCount;
        uint64_t flushSeq{0};
    };

    using DeserializeAndWriteFn = void (*)(ByteBuffer &buf, int rowCount,
                                          sqlite3_int64 idOffset, int nodeId,
                                          WriterBackend *backend);

    struct WriterChannel {
        WriterBackend *backend{nullptr};
        DeserializeAndWriteFn deserializeAndWrite{nullptr};
        std::queue<BatchItem> queue;
        std::mutex mutex;
        std::condition_variable cv;
        // 'worker' and 'done' describe writer liveness.  Both are written
        // under 'mutex' so flush() can safely decide whether the channel can
        // still service a flush request.
        std::thread *worker{nullptr};
        bool done{false};
        std::mutex flushMutex;
        std::condition_variable flushCv;
        uint64_t nextFlushSeq{0};
        // Written and read only under 'flushMutex'.
        uint64_t completedFlushSeq{0};
    };

    // Upper bound on how long flush() will block waiting for a writer thread.
    // A lost wakeup then degrades to a slow flush instead of a hung process.
    static const int FLUSH_WAIT_TIMEOUT_MS = 5000;

    void registerChannel(const char *tag, bool directWrite, DeserializeAndWriteFn fn);
    void acceptLoop();
    void recvLoop(TcpConnection *conn, WriterChannel *channel);
    static void writerLoop(WriterChannel *channel);
    static void signalFlushComplete(WriterChannel *channel, uint64_t flushSeq);

    TcpConnection m_listener;
    std::thread *m_acceptThread{nullptr};
    std::unordered_map<std::string, WriterChannel*> m_channels;
    std::vector<std::thread*> m_recvThreads;
    std::vector<TcpConnection*> m_recvConns;
    std::mutex m_connMutex;

    DbResource *m_resource{nullptr};
    int m_port{0};
    std::string m_basefile;
    bool m_directWrite{false};
    std::atomic<bool> m_running{false};
};

}  // namespace rpdtracer
