#pragma once

#include <string>
#include <map>
#include <unordered_map>

#include <roctracer_hip.h>

// Usage
//   contains() are items you are interested in, i.e. matches the filter
//   "normal mode": things you add() are the only things matching the filter
//   invertMode() == true: All things match filter except what you add()

class ApiIdList
{
public:
  ApiIdList();
  bool invertMode() { return m_invert; }
  void setInvertMode(bool invert) { m_invert = invert; }
  void add(std::string apiName);
  void remove(std::string apiName);
  bool loadUserPrefs();

  bool contains(uint32_t apiId);

  const std::unordered_map<uint32_t, uint32_t> &filterList() { return m_filter; }

private:
  std::map<std::string, uint32_t> m_ids;		// api string -> apiId
  std::unordered_map<uint32_t, uint32_t> m_filter;	// apiId -> "1"
  bool m_invert;

  void loadApiNames();
};

