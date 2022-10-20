/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

//#include "Logger.h"

class DataSource
{
public:
    //DataSource();
    virtual void init() = 0;
    virtual void end() = 0;
    virtual void startTracing() = 0;
    virtual void stopTracing() = 0;
};
