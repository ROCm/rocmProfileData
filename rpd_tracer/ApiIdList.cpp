/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include "ApiIdList.h"

//#include <roctracer_hip.h>
// FIXME: make this work for cud and hip or turn into interface

ApiIdList::ApiIdList()
: m_invert(true)
{
}

void ApiIdList::add(std::string apiName)
{
#if 0
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
    m_filter[cid] = 1;
#endif
}
void ApiIdList::remove(std::string apiName)
{
#if 0
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
    m_filter.erase(cid);
#endif
}

bool ApiIdList::loadUserPrefs()
{
  // FIXME: check an ENV variable that points to an exclude file
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (m_filter.find(apiId) != m_filter.end()) ? !m_invert : m_invert;  // XOR
}
