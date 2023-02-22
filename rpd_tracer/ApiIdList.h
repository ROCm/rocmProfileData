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

