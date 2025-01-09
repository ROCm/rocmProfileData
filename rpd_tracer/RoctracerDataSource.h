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
#pragma once

#include <roctracer.h>

#include <string>
#include <cstddef>
#include <cstdint>

#include "DataSource.h"
#include "ApiIdList.h"
#include "Logger.h"

class RocmApiIdList : public ApiIdList
{
public:
    RocmApiIdList() { ; }
    uint32_t mapName(const std::string &apiName) override;
};


class RoctracerDataSource : public DataSource
{
public:
    //RoctracerDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;

private:
    RocmApiIdList m_apiList;

    roctracer_pool_t *m_hccPool{nullptr};
    static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
    static void hcc_activity_callback(const char* begin, const char* end, void* arg);
    static inline int unwind(Logger &logger, const char* api, const sqlite_int64 api_id);

};
