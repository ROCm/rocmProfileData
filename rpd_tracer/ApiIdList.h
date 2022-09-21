/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <string>
#include <map>
#include <unordered_map>

//#include <roctracer_hip.h>

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
  void add(const std::string &apiName);
  void remove(const std::string &apiName);
  bool loadUserPrefs();

  // Map api string to cnid enum
  virtual uint32_t mapName(const std::string &apiName) = 0;

  bool contains(uint32_t apiId);

  const std::unordered_map<uint32_t, uint32_t> &filterList() { return m_filter; }

private:
  std::unordered_map<uint32_t, uint32_t> m_filter;	// apiId -> "1"
  bool m_invert;
};

