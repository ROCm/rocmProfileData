# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(rocmProfileData LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locates a ROCm install, whether it's the traditional /opt/rocm, an
# explicit ROCM_PATH/ROCM_HOME, or a python-venv based install (e.g.
# TheRock's rocm-sdk wheels).  Sets ROCM_ROOT_DIR and augments
# CMAKE_PREFIX_PATH accordingly.
include(FindROCmRoot)

find_package(Python3 COMPONENTS Interpreter)

# Enabled unconditionally (rather than only when RPD_BUILD_TESTS is ON) so
# that `ctest` run from the top-level build directory discovers all tests,
# including rlog's own (rlog/CMakeLists.txt calls enable_testing() itself,
# but that only takes effect for directories below it, not the whole tree).
enable_testing()

option(RPD_BUILD_TRACER "Build the native rpd_tracer/rpd_embedded LD_PRELOAD/embedded libraries" ON)
option(RPD_BUILD_REMOTE "Build the librpd_remote remote start/stop helper" ON)
option(RPD_BUILD_PYTHON "Install the rocpd_python schema/import/analysis package" ON)
option(RPD_BUILD_VIEWER "Install the rpd_viewer (rpd-viewer) web viewer package" ON)

# rlog is a mandatory build dependency of rpd_tracer (RlogClient.cpp,
# RlogDataSource.cpp). It is vendored as a self-referencing git submodule
# (same repo, "rlog" branch) rather than an external project.
if(RPD_BUILD_TRACER)
  set(_rlog_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/rlog")
  if(NOT EXISTS "${_rlog_src_dir}/CMakeLists.txt")
    message(FATAL_ERROR
      "RPD_BUILD_TRACER is ON but the rlog submodule is not present.\n"
      "Run: git submodule update --init rlog")
  endif()
  add_subdirectory(rlog)
endif()

if(RPD_BUILD_TRACER)
  add_subdirectory(rpd_tracer)
endif()

if(RPD_BUILD_REMOTE)
  add_subdirectory(remote)
endif()

if(RPD_BUILD_PYTHON)
  if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "RPD_BUILD_PYTHON is ON but no python3 interpreter was found")
  endif()
  add_custom_target(rocpd_python ALL
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rocpd_python"
    COMMENT "Installing rocpd_python (schema / importers / analysis helpers)"
    VERBATIM
  )
  install(CODE "execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}/rocpd_python\"
    COMMAND_ERROR_IS_FATAL ANY
  )")
endif()

if(RPD_BUILD_VIEWER)
  if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "RPD_BUILD_VIEWER is ON but no python3 interpreter was found")
  endif()
  add_custom_target(rpd_viewer ALL
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rpd_viewer"
    COMMENT "Installing rpd_viewer (rpd-viewer web trace viewer)"
    VERBATIM
  )
  if(RPD_BUILD_PYTHON)
    add_dependencies(rpd_viewer rocpd_python)
  endif()
  install(CODE "execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}/rpd_viewer\"
    COMMAND_ERROR_IS_FATAL ANY
  )")
endif()

message(STATUS "")
message(STATUS "rocmProfileData configuration summary:")
message(STATUS "  RPD_BUILD_TRACER            = ${RPD_BUILD_TRACER}")
message(STATUS "  RPD_BUILD_REMOTE            = ${RPD_BUILD_REMOTE}")
message(STATUS "  RPD_BUILD_PYTHON            = ${RPD_BUILD_PYTHON}")
message(STATUS "  RPD_BUILD_VIEWER            = ${RPD_BUILD_VIEWER}")
message(STATUS "  RPD_ENABLE_CPPTRACE         = ${RPD_ENABLE_CPPTRACE}")
message(STATUS "  RPD_ENABLE_ROCM_TRACE_LITE  = ${RPD_ENABLE_ROCM_TRACE_LITE}")
message(STATUS "  ROCM_ROOT_DIR               = ${ROCM_ROOT_DIR}")
message(STATUS "")
