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
  # lib/rocm_sysdeps exists only in rocm-sdk/TheRock wheel layouts; use it
  # to tell a python-venv install apart from a traditional (ldconfig-
  # visible) one.  ROCM_ROOT_HAS_SYSDEPS is consumed by the top-level
  # RPD_INSTALL_RPATH option.
  if(EXISTS "${ROCM_ROOT_DIR}/lib/rocm_sysdeps")
    set(ROCM_ROOT_HAS_SYSDEPS TRUE)
  else()
    set(ROCM_ROOT_HAS_SYSDEPS FALSE)
  endif()
  message(STATUS "ROCm root: ${ROCM_ROOT_DIR} (rocm_sysdeps: ${ROCM_ROOT_HAS_SYSDEPS})")
else()
  message(STATUS "ROCm root: not found (searched ROCM_PATH/ROCM_HOME, python venv, /opt/rocm)")
endif()

#[[
rocm_find_package(<args...>) like find_package(<args...>), but with
CMAKE_PREFIX_PATH/CMAKE_INCLUDE_PATH/CMAKE_LIBRARY_PATH temporarily
augmented with the ROCm install's own cmake config dirs and (for
rocm-sdk/TheRock devel wheels) its vendored third-party "sysdeps" tree
(lib/rocm_sysdeps sqlite3, zstd, zlib, bz2, numa, libelf, liblzma, etc).

This augmentation is scoped to the single find_package() call made here,
NOT set globally, on purpose: rocm-sdk devel wheels vendor their own
private copies of common system libraries under lib/rocm_sysdeps for
ROCm's internal use (e.g. MIOpen's sqlite3 cache), and those must never
be allowed to silently satisfy an unrelated find_package(Foo) elsewhere
in the project (or its subdirectories, e.g. rlog) that actually wants the
system's Foo. Only call rocm_find_package() for packages that are
genuinely part of the ROCm SDK itself (hip, rocprofiler-sdk,
hsa-runtime64, nlohmann_json, ...); use plain find_package() for
everything else.
#]]
macro(rocm_find_package)
  if(ROCM_ROOT_DIR)
    set(_rocm_fp_saved_prefix_path "${CMAKE_PREFIX_PATH}")
    set(_rocm_fp_saved_include_path "${CMAKE_INCLUDE_PATH}")
    set(_rocm_fp_saved_library_path "${CMAKE_LIBRARY_PATH}")

    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT_DIR}" "${ROCM_ROOT_DIR}/lib/cmake")
    if(EXISTS "${ROCM_ROOT_DIR}/lib/rocm_sysdeps")
      list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps")
      list(APPEND CMAKE_INCLUDE_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps/include")
      list(APPEND CMAKE_LIBRARY_PATH "${ROCM_ROOT_DIR}/lib/rocm_sysdeps/lib")
    endif()

    find_package(${ARGV})

    set(CMAKE_PREFIX_PATH "${_rocm_fp_saved_prefix_path}")
    set(CMAKE_INCLUDE_PATH "${_rocm_fp_saved_include_path}")
    set(CMAKE_LIBRARY_PATH "${_rocm_fp_saved_library_path}")
    unset(_rocm_fp_saved_prefix_path)
    unset(_rocm_fp_saved_include_path)
    unset(_rocm_fp_saved_library_path)
  else()
    find_package(${ARGV})
  endif()
endmacro()
