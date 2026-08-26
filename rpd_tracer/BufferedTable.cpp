/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Table.h"
#include "Utility.h"

#include <thread>

using rpdtracer::BufferedTable;

namespace rpdtracer {

class BufferedTablePrivate
{
public:
    BufferedTablePrivate(BufferedTable *cls) : p(cls) {}

    void work();                // work thread
    std::thread *worker;
    bool done;
    bool workerRunning;
    bool flushRequested;
    int flushTarget;

    BufferedTable *p;
};

BufferedTable::BufferedTable(const char *basefile, int bufferSize, int batchsize)
: Table(basefile)
, BUFFERSIZE(bufferSize)
, BATCHSIZE(batchsize)
, d(new BufferedTablePrivate(this))
{
    d->done = false;
    d->workerRunning = true;
    d->flushRequested = false;
    d->flushTarget = 0;
    d->worker = new std::thread(&BufferedTablePrivate::work, d);
}

BufferedTable::~BufferedTable()
{
    delete d;
    // finalize here?  Possibly a second time
}


void BufferedTable::flush()
{
    std::unique_lock<std::mutex> lock(m_mutex);

    d->flushTarget = m_head;
    d->flushRequested = true;
    m_wait.notify_one();
    while (d->flushRequested)
        m_wait.wait(lock);

    // Table specific flush
    flushRows();	// While holding m_mutex
}


void BufferedTable::finalize()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    d->done = true;
    m_wait.notify_one();
    lock.unlock();
    d->worker->join();
    d->workerRunning = false;
    delete d->worker;

    flushRows();
}


bool BufferedTable::workerRunning()
{
    return d->workerRunning;
}

void BufferedTablePrivate::work()
{
    std::unique_lock<std::mutex> lock(p->m_mutex);

    while (done == false) {
        while ((p->m_head - p->m_tail) >= p->BATCHSIZE) {
            lock.unlock();
            p->writeRows();
            p->m_wait.notify_all();
            lock.lock();
        }
        if (flushRequested) {
            while (p->m_tail < flushTarget) {
                lock.unlock();
                p->writeRows();
                lock.lock();
            }
            flushRequested = false;
            p->m_wait.notify_all();
        }
        workerRunning = false;
        if (done == false)
            p->m_wait.wait(lock);
        workerRunning = true;
    }
    // done: drain remaining rows before exit
    while (p->m_head > p->m_tail) {
        lock.unlock();
        p->writeRows();
        lock.lock();
    }
}

}  // namespace rpdtracer
