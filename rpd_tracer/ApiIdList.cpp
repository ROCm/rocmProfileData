
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
#include "ApiIdList.h"

//#include <roctracer_hip.h>
// FIXME: make this work for cud and hip or turn into interface

ApiIdList::ApiIdList()
: m_invert(true)
{
}

void ApiIdList::add(const std::string &apiName)
{
    uint32_t cid = mapName(apiName);
    if (cid > 0)
        m_filter[cid] = 1;
#if 0
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
    m_filter[cid] = 1;
#endif
}
void ApiIdList::remove(const std::string &apiName)
{
    uint32_t cid = mapName(apiName);
    if (cid > 0)
        m_filter.erase(cid);
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
