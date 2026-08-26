/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "TcpConnection.h"
#include "Utility.h"

#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/tcp.h>

namespace rpdtracer {

TcpConnection::TcpConnection() = default;

TcpConnection::TcpConnection(int fd)
: m_fd(fd)
{
}

TcpConnection::~TcpConnection()
{
    close();
}

bool TcpConnection::connect(const char *host, int port)
{
    struct addrinfo hints{}, *result;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char portStr[16];
    snprintf(portStr, sizeof(portStr), "%d", port);

    if (getaddrinfo(host, portStr, &hints, &result) != 0) {
        rpdLog("TcpConnection: getaddrinfo failed for %s:%d: %s\n", host, port, strerror(errno));
        return false;
    }

    m_fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (m_fd < 0) {
        rpdLog("TcpConnection: socket failed: %s\n", strerror(errno));
        freeaddrinfo(result);
        return false;
    }

    int yes = 1;
    setsockopt(m_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    if (::connect(m_fd, result->ai_addr, result->ai_addrlen) != 0) {
        rpdLog("TcpConnection: connect to %s:%d failed: %s\n", host, port, strerror(errno));
        ::close(m_fd);
        m_fd = -1;
        freeaddrinfo(result);
        return false;
    }

    freeaddrinfo(result);
    return true;
}

bool TcpConnection::listen(int port)
{
    m_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_fd < 0) {
        rpdLog("TcpConnection: socket failed: %s\n", strerror(errno));
        return false;
    }

    int yes = 1;
    setsockopt(m_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(m_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        rpdLog("TcpConnection: bind to port %d failed: %s\n", port, strerror(errno));
        ::close(m_fd);
        m_fd = -1;
        return false;
    }

    if (::listen(m_fd, 128) != 0) {
        rpdLog("TcpConnection: listen failed: %s\n", strerror(errno));
        ::close(m_fd);
        m_fd = -1;
        return false;
    }

    return true;
}

TcpConnection* TcpConnection::accept()
{
    struct sockaddr_in clientAddr{};
    socklen_t addrLen = sizeof(clientAddr);

    int clientFd = ::accept(m_fd, reinterpret_cast<struct sockaddr*>(&clientAddr), &addrLen);
    if (clientFd < 0)
        return nullptr;

    int yes = 1;
    setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    return new TcpConnection(clientFd);
}

bool TcpConnection::send(const void *data, size_t len)
{
    const char *p = static_cast<const char*>(data);
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = ::send(m_fd, p, remaining, MSG_NOSIGNAL);
        if (n <= 0)
            return false;
        p += n;
        remaining -= n;
    }
    return true;
}

bool TcpConnection::recv(void *data, size_t len)
{
    char *p = static_cast<char*>(data);
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = ::recv(m_fd, p, remaining, 0);
        if (n <= 0)
            return false;
        p += n;
        remaining -= n;
    }
    return true;
}

void TcpConnection::close()
{
    if (m_fd >= 0) {
        ::shutdown(m_fd, SHUT_RDWR);
        ::close(m_fd);
        m_fd = -1;
    }
}

}  // namespace rpdtracer
