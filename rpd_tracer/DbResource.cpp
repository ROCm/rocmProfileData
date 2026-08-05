// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "DbResource.h"

#include <fmt/format.h>

using rpdtracer::DbResource;

namespace rpdtracer {

class DbResourcePrivate
{
public:
    DbResourcePrivate(DbResource *cls) : p(cls) {}

    sqlite3 *connection;
    std::string resourceName;

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

}  // namespace rpdtracer
