// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

//#include <roctracer.h>

#include <string>

#include <rocprofiler-sdk/registration.h>

#include "DataSource.h"
#include "ApiIdList.h"

namespace rpdtracer {

class RocprofDataSourcePrivate;
class RocprofDataSource : public DataSource
{
public:
    RocprofDataSource();
    ~RocprofDataSource();
    void init() override;
    void end() override;
    void startTracing() override;
    void stopTracing() override;
    void flush() override;

private:
    RocprofDataSourcePrivate *d;
    friend class RocprofDataSourcePrivate;

    //RocmApiIdList m_apiList;

public:
      static int toolInit(rocprofiler_client_finalize_t finalize_func, void* tool_data);
      static void toolFinialize(void* tool_data);

      static void api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data);
      static void buffer_callback(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id, rocprofiler_record_header_t** headers, size_t num_headers, void* user_data, uint64_t drop_count);
      static void code_object_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data);

};

}    // namespace rpdtracer
