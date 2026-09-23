#[[
Copyright (C) Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT

Locate a ROCm installation and expose it as ROCM_ROOT_DIR.

ROCm can be installed in a few different places depending on how it was
provisioned:

  1. The traditional system package location, /opt/rocm (optionally
     versioned, e.g. /opt/rocm-6.2.0, with /opt/rocm as a symlink).
  2. Exported explicitly via the ROCM_PATH or ROCM_HOME environment
     variables (used by ROCm's own build tooling).
  3. Inside a python virtual environment, via "TheRock"/rocm-sdk wheel
     installs (e.g. rocm-sdk-devel).  These place a full ROCm SDK under
     <venv>/lib/pythonX.Y/site-packages/_rocm_sdk_devel rather than
     /opt/rocm.  When such a venv is active, ROCM_PATH/ROCM_HOME are
     typically already exported by the venv's activation hooks, but we
     also probe for the package directly in case they are not.

Resolution order: explicit CMake cache var > environment variables >
active python venv site-packages > /opt/rocm.
#]]

if(DEFINED ROCM_ROOT_DIR AND ROCM_ROOT_DIR)
  set(_rocm_root_hint "${ROCM_ROOT_DIR}")
elseif(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  set(_rocm_root_hint "$ENV{ROCM_PATH}")
elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
  set(_rocm_root_hint "$ENV{ROCM_HOME}")
else()
  set(_rocm_root_hint "")
endif()

# Probe a venv (or any python3 on PATH) for a rocm_sdk / _rocm_sdk_devel
# installation, e.g. TheRock's rocm-sdk-devel wheel.
function(_rocm_probe_python_sdk out_var)
  set(${out_var} "" PARENT_SCOPE)
  find_program(_rocm_python_exe NAMES python3 python)
  if(NOT _rocm_python_exe)
    return()
  endif()
  execute_process(
    COMMAND "${_rocm_python_exe}" -c
      "import importlib.util as u; s = u.find_spec('_rocm_sdk_devel'); print(s.submodule_search_locations[0] if s else '')"
    OUTPUT_VARIABLE _rocm_sdk_devel_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_rocm_sdk_devel_dir AND EXISTS "${_rocm_sdk_devel_dir}")
    set(${out_var} "${_rocm_sdk_devel_dir}" PARENT_SCOPE)
  endif()
endfunction()

if(NOT _rocm_root_hint)
  _rocm_probe_python_sdk(_rocm_sdk_devel_dir)
  if(_rocm_sdk_devel_dir)
    set(_rocm_root_hint "${_rocm_sdk_devel_dir}")
  endif()
endif()

if(NOT _rocm_root_hint AND EXISTS "/opt/rocm")
  set(_rocm_root_hint "/opt/rocm")
endif()

if(_rocm_root_hint)
  get_filename_component(ROCM_ROOT_DIR "${_rocm_root_hint}" ABSOLUTE)
else()
  set(ROCM_ROOT_DIR "")
endif()

if(ROCM_ROOT_DIR)
  set(CMAKE_PREFIX_PATH "${ROCM_ROOT_DIR};${ROCM_ROOT_DIR}/lib/cmake;${CMAKE_PREFIX_PATH}")
  # rocm-sdk (TheRock) devel wheels keep third-party sysdeps (sqlite3, zstd,
  # fmt, etc) under lib/rocm_sysdeps rather than the top level lib/include.
  if(EXISTS "${ROCM_ROOT_DIR}/lib/rocm_sysdeps")
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps")
    list(APPEND CMAKE_INCLUDE_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps/include")
    list(APPEND CMAKE_LIBRARY_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps/lib")
  endif()
  message(STATUS "ROCm root: ${ROCM_ROOT_DIR}")
else()
  message(STATUS "ROCm root: not found (searched ROCM_PATH/ROCM_HOME, python venv, /opt/rocm)")
endif()
