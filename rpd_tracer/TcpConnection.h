/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <cstddef>

namespace rpdtracer {

class TcpConnection {
public:
    TcpConnection();
    ~TcpConnection();

    TcpConnection(const TcpConnection&) = delete;
    TcpConnection& operator=(const TcpConnection&) = delete;

    // Client: connect to a remote host:port
    bool connect(const char *host, int port);

    // Server: bind and listen on a port
    bool listen(int port);

    // Server: accept a new connection (blocks until a client connects)
    // Caller owns the returned pointer.
    TcpConnection* accept();

    // Send exactly len bytes. Returns false on error/disconnect.
    bool send(const void *data, size_t len);

    // Receive exactly len bytes. Returns false on error/disconnect.
    bool recv(void *data, size_t len);

    void close();
    bool isConnected() const { return m_fd >= 0; }
    int fd() const { return m_fd; }

private:
    TcpConnection(int fd);
    int m_fd{-1};
};

}  // namespace rpdtracer
