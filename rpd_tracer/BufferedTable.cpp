/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Table.h"
#include "Utility.h"

#include <thread>


class BufferedTablePrivate
{
public:
    BufferedTablePrivate(BufferedTable *cls) : p(cls) {}

    void work();                // work thread
    std::thread *worker;
    bool done;
    bool workerRunning;

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

    // wait for worker to pause
    while (d->workerRunning == true)
        m_wait.wait(lock);

    // Worker paused, clear the buffer ourselves
    auto flushPoint = m_head;
    while (flushPoint > m_tail) {
        lock.unlock();
        writeRows();
        lock.lock();
    }

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

    flush();
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
        workerRunning = false;
        if (done == false)
            p->m_wait.wait(lock);
        workerRunning = true;
    }
}

