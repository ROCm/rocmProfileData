/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "Utility.h"

#include <cstdlib>
#include <map>
#include <string>

namespace rpdtracer {

std::map<std::string, std::string>& configMap()
{
    static std::map<std::string, std::string> s_apiValues;
    return s_apiValues;
}

void setConfig(const char *property, const char *value)
{
    configMap()[property] = value;
}

const char* getConfig(const char *envvar, const char *property, const char *defaultValue)
{
    auto &m = configMap();
    auto it = m.find(property);
    if (it != m.end())
        return it->second.c_str();
    const char *val = std::getenv(envvar);
    if (val != nullptr)
        return val;
    return rlog::getProperty("rpd_tracer", property, defaultValue);
}

}    // namespace rpdtracer
