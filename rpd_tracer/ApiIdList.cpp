#include "ApiIdList.h"

ApiIdList::ApiIdList()
: m_invert(true)
{
  loadApiNames();
}

void ApiIdList::add(std::string apiName)
{
  auto it = m_ids.find(apiName);
  if (it != m_ids.end())
    m_filter[it->second] = 1;
}
void ApiIdList::remove(std::string apiName)
{
  auto it = m_ids.find(apiName);
  if (it != m_ids.end())
    m_filter.erase(it->second);
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

void ApiIdList::loadApiNames()
{
  // Build lut from apiName to apiId
  for (uint32_t i = 0; i < HIP_API_ID_NUMBER; ++i) {
    m_ids[std::string(hip_api_name(i))] = i;
  }
}

