/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "rlog/client.h"

namespace rpdtracer {

void rlogClientInit()
{
    rlog::init();
    rlog::setDefaultDomain("rpd_tracer");
}

}  // namespace rpdtracer
