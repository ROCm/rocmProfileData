/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <sqlite3.h>
#include <mutex>
#include <condition_variable>

class DbResourcePrivate;
class DbResource
{
public:
    DbResource(const std::string &basefile, const std::string &resourceName);
    ~DbResource();

    void lock();
    bool tryLock();
    void unlock();

    bool isLocked();

private:
    DbResourcePrivate *d;
    friend class DbResourcePrivate;
};
