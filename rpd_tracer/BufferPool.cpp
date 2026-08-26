/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "BufferPool.h"
#include "Utility.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static const size_t DEFAULT_SHM_SIZE = 512UL * 1024 * 1024;

namespace rpdtracer {

BufferPool::BufferPool() = default;

BufferPool::~BufferPool()
{
    // Destruct row elements owned by this process
    for (auto it = m_entries.rbegin(); it != m_entries.rend(); ++it) {
        it->destructor();
        delete it->slot;
        if (!m_shared)
            std::free(it->memory);
    }
    m_entries.clear();

    if (m_shared && m_shmBase != nullptr) {
        PoolHeader *pool = static_cast<PoolHeader*>(m_shmBase);
        munmap(m_shmBase, pool->totalSize);
        if (m_shmFd >= 0)
            close(m_shmFd);
    }
}

void BufferPool::initShared(const char *dbFilename)
{
    // Derive shm name from db filename
    std::string base(dbFilename);
    auto pos = base.rfind('/');
    if (pos != std::string::npos)
        base = base.substr(pos + 1);
    m_shmName = "/rpd_" + base;

    const char *sizeEnv = getenv("RPDT_SHM_SIZE");
    size_t totalSize = sizeEnv ? strtoul(sizeEnv, nullptr, 0) : DEFAULT_SHM_SIZE;

    // Try to be the creator
    int fd = shm_open(m_shmName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd >= 0) {
        // Creator path
        m_creator = true;
        if (ftruncate(fd, totalSize) != 0) {
            rpdLog("BufferPool: ftruncate failed: %s\n", strerror(errno));
            close(fd);
            shm_unlink(m_shmName.c_str());
            return;
        }
        void *base = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            rpdLog("BufferPool: mmap failed: %s\n", strerror(errno));
            close(fd);
            shm_unlink(m_shmName.c_str());
            return;
        }
        m_shmBase = base;
        m_shmFd = fd;

        PoolHeader *pool = static_cast<PoolHeader*>(m_shmBase);
        pool->totalSize = totalSize;
        pool->allocOffset.store(sizeof(PoolHeader), std::memory_order_relaxed);
        pool->slotCount.store(0, std::memory_order_relaxed);
        pool->ready.store(1, std::memory_order_release);

        m_shared = true;
        rpdLog("BufferPool: created shm %s (%zu bytes)\n", m_shmName.c_str(), totalSize);

    } else if (errno == EEXIST) {
        // Joiner path
        fd = shm_open(m_shmName.c_str(), O_RDWR, 0);
        if (fd < 0) {
            // Stale shm — unlink and retry as creator
            shm_unlink(m_shmName.c_str());
            fd = shm_open(m_shmName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd < 0) {
                rpdLog("BufferPool: shm_open retry failed: %s\n", strerror(errno));
                return;
            }
            // Recurse into creator path
            m_creator = true;
            if (ftruncate(fd, totalSize) != 0) {
                rpdLog("BufferPool: ftruncate failed: %s\n", strerror(errno));
                close(fd);
                shm_unlink(m_shmName.c_str());
                return;
            }
            void *base = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) {
                rpdLog("BufferPool: mmap failed: %s\n", strerror(errno));
                close(fd);
                shm_unlink(m_shmName.c_str());
                return;
            }
            m_shmBase = base;
            m_shmFd = fd;

            PoolHeader *pool = static_cast<PoolHeader*>(m_shmBase);
            pool->totalSize = totalSize;
            pool->allocOffset.store(sizeof(PoolHeader), std::memory_order_relaxed);
            pool->slotCount.store(0, std::memory_order_relaxed);
            pool->ready.store(1, std::memory_order_release);

            m_shared = true;
            rpdLog("BufferPool: created shm %s after stale cleanup (%zu bytes)\n", m_shmName.c_str(), totalSize);
            return;
        }

        // Need to know the size before mmap — read from the first page
        void *base = mmap(nullptr, sizeof(PoolHeader), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            rpdLog("BufferPool: mmap header failed: %s\n", strerror(errno));
            close(fd);
            return;
        }

        PoolHeader *pool = static_cast<PoolHeader*>(base);
        // Spin until creator sets ready flag
        while (pool->ready.load(std::memory_order_acquire) == 0)
            usleep(100);

        size_t actualSize = pool->totalSize;
        munmap(base, sizeof(PoolHeader));

        // Remap with full size
        base = mmap(nullptr, actualSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            rpdLog("BufferPool: mmap full failed: %s\n", strerror(errno));
            close(fd);
            return;
        }

        m_shmBase = base;
        m_shmFd = fd;
        m_shared = true;
        rpdLog("BufferPool: joined shm %s (%zu bytes)\n", m_shmName.c_str(), actualSize);
    } else {
        rpdLog("BufferPool: shm_open failed: %s\n", strerror(errno));
    }
}

}  // namespace rpdtracer
