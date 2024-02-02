/*********************************************************************************
* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
********************************************************************************/
#include "DbResource.h"

#include <fmt/format.h>

class DbResourcePrivate
{
public:
    DbResourcePrivate(DbResource *cls) : p(cls) {}

    sqlite3 *connection;
    std::string resourceName;

    //std::mutex m_mutex;
    //std::condition_variable m_wait;

    bool locked {false};

    DbResource *p;

    static int resourceCallback(void *data, int argc, char **argv, char **colName);
};


DbResource::DbResource(const std::string &basefile, const std::string &resourceName)
: d(new DbResourcePrivate(this))
{
    sqlite3_open(basefile.c_str(), &d->connection);
    d->resourceName = resourceName;
}

DbResource::~DbResource()
{
    unlock();
    sqlite3_close(d->connection);
}

int DbResourcePrivate::resourceCallback(void *data, int argc, char **argv, char **colName)
{
    sqlite3_int64 &resourceId = *(sqlite3_int64*)data;
    resourceId = atoll(argv[0]);
    return 0;
}

void DbResource::lock()
{
}

bool DbResource::tryLock()
{
   if (d->locked == false) {
       // check if available
       int ret;
       char *error_msg;

       sqlite3_int64 resourceValue = -1;
       ret = sqlite3_exec(d->connection, fmt::format("SELECT value FROM rocpd_metadata WHERE tag = 'resourceLock::{}'", d->resourceName).c_str(), &DbResourcePrivate::resourceCallback, &resourceValue, &error_msg);
       if (resourceValue <= 0) {
           // Not locked.  Lock db and look again
           sqlite3_exec(d->connection, "BEGIN EXCLUSIVE TRANSACTION", NULL, NULL, NULL);
           resourceValue = -1;
           ret = sqlite3_exec(d->connection, fmt::format("SELECT value FROM rocpd_metadata WHERE tag = 'resourceLock::{}'", d->resourceName).c_str(), &DbResourcePrivate::resourceCallback, &resourceValue, &error_msg);
           if (resourceValue == -1) {
               // Not initialize, "make and take"
               ret = sqlite3_exec(d->connection, fmt::format("INSERT into rocpd_metadata(tag, value) VALUES ('resourceLock::{}', 1)", d->resourceName).c_str(), NULL, NULL, &error_msg);
               if (ret == SQLITE_OK)
                   d->locked = true;
           }          
           else if (resourceValue == 0) {
               // take resource
               ret = sqlite3_exec(d->connection, fmt::format("UPDATE rocpd_metadata SET value = '1' WHERE tag = 'resourceLock::{}'", d->resourceName).c_str(), NULL, NULL, &error_msg);
               if (ret == SQLITE_OK)
                   d->locked = true;
           }
           sqlite3_exec(d->connection, "END TRANSACTION", NULL, NULL, NULL);
       }
   }
   return d->locked; 
}

void DbResource::unlock()
{
    if (d->locked) {
        int ret;
        char *error_msg;
        sqlite3_exec(d->connection, "BEGIN EXCLUSIVE TRANSACTION", NULL, NULL, NULL);
        ret = sqlite3_exec(d->connection, fmt::format("UPDATE rocpd_metadata SET value = '0' WHERE tag = 'resourceLock::{}'", d->resourceName).c_str(), NULL, NULL, &error_msg);
        sqlite3_exec(d->connection, "END TRANSACTION", NULL, NULL, NULL);
    }
}

bool DbResource::isLocked()
{
    return d->locked;
}
