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
        std::thread *worker{nullptr};
        bool done{false};
    };

    void registerChannel(const char *tag, bool directWrite, DeserializeAndWriteFn fn);
    void acceptLoop();
    void recvLoop(TcpConnection *conn, WriterChannel *channel);
    static void writerLoop(WriterChannel *channel);

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
    bool m_running{false};
};

}  // namespace rpdtracer
