# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Design note on GPU activity backends
# -------------------------------------
# rpd_tracer does NOT pick a single build-time "backend". Every GPU activity
# DataSource whose headers/libs are detected below (ClrDataSource,
# RocprofDataSource, RoctracerDataSource, RtlDataSource, CuptiDataSource) is
# compiled into the resulting libraries. At runtime, Logger::init() (see
# Logger.cpp) dlsym's each DataSource's factory function and instantiates
# whichever ones are present, picking exactly one of the mutually-exclusive
# "HIP data provider" family (Clr/Rocprof/Roctracer/Rtl/Cupti) via a priority
# list. That runtime choice is controlled entirely by environment variables /
# rlog properties (RPDT_DATASOURCES_PRIORITY, RPDT_DATASOURCES_EXPLICIT,
# RPDT_DATASOURCES_EXCLUDE) — see docs/README_embedded.md. There is no
# build-time "which backend" CMake option; the closest thing to one is
# disabling detection of a given ROCm component entirely (not currently
# exposed as an option, since the underlying Makefile doesn't expose it
# either).

find_package(SQLite3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Threads REQUIRED)

set(RPD_COMMON_SRCS
  Schema.cpp
  Table.cpp
  BufferedTable.cpp
  OpTable.cpp
  KernelApiTable.cpp
  CopyApiTable.cpp
  ApiTable.cpp
  StringTable.cpp
  UStringTable.cpp
  MetadataTable.cpp
  MonitorTable.cpp
  StackFrameTable.cpp
  CounterTable.cpp
  ApiIdList.cpp
  DbResource.cpp
  Config.cpp
  Storage.cpp
  Logger.cpp
  Unwind.cpp
  RoctxDataSource.cpp
  NvtxDataSource.cpp
  RlogClient.cpp
  ChronoSyncDataSource.cpp
  Firefly.cpp
  Utility.cpp
  TcpConnection.cpp
  NetWriterBackend.cpp
  RemoteDataSource.cpp
)

# ---------------------------------------------------------------------------
# Common OBJECT library: everything shared between librpd_tracer.so and
# librpd_embedded.so. Each final target additionally links its own compiled
# variant of rpd_capi.cpp (see bottom of this file).
# ---------------------------------------------------------------------------
add_library(rpd_tracer_objs OBJECT ${RPD_COMMON_SRCS})
set_target_properties(rpd_tracer_objs PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(rpd_tracer_objs PRIVATE AMD_INTERNAL_BUILD)
target_link_libraries(rpd_tracer_objs PUBLIC SQLite::SQLite3 fmt::fmt Threads::Threads)

# rlog: mandatory dependency, used two different ways (mirrors the Makefile,
# which both compiles rlog/src/rlog.cpp directly into rpd_tracer AND links
# -lrlog):
#  - rlog/src/rlog.cpp is a dlopen-based client shim (rlog::mark/init/etc,
#    used by RlogClient.cpp) that dynamically loads librlog.so at runtime
#    via dlsym rather than linking against it.
#  - RlogDataSource.cpp uses rlog::Hub directly (rlog::Hub::singleton()),
#    which requires linking against the rlog_hub target itself (also pulls
#    in its include dirs: include/ for client.h/Logger.h, src/ for Hub.h).
target_sources(rpd_tracer_objs PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../rlog/src/rlog.cpp"
  RlogDataSource.cpp
)
target_link_libraries(rpd_tracer_objs PUBLIC rlog_hub ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Generated schema headers (baked-in copies of rocpd_python's sql schema)
# ---------------------------------------------------------------------------
find_program(XXD_EXECUTABLE xxd REQUIRED)

set(RPD_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rocpd_python/rocpd/schema_data")

function(rpd_generate_schema_header name)
  set(_src "${RPD_SCHEMA_DIR}/${name}.cmd")
  set(_out "${CMAKE_CURRENT_BINARY_DIR}/${name}.h")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND ${CMAKE_COMMAND} -E echo "static const unsigned char ${name}_cmd[] = {" > "${_out}"
    COMMAND ${XXD_EXECUTABLE} -i < "${_src}" >> "${_out}"
    COMMAND ${CMAKE_COMMAND} -E echo ", 0x00" >> "${_out}"
    COMMAND ${CMAKE_COMMAND} -E echo "};" >> "${_out}"
    DEPENDS "${_src}"
    VERBATIM
  )
  set(${name}_HEADER "${_out}" PARENT_SCOPE)
endfunction()

rpd_generate_schema_header(tableSchema)
rpd_generate_schema_header(utilitySchema)

add_custom_target(rpd_tracer_schema_headers DEPENDS
  "${tableSchema_HEADER}"
  "${utilitySchema_HEADER}"
)
add_dependencies(rpd_tracer_objs rpd_tracer_schema_headers)
target_include_directories(rpd_tracer_objs PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
set_source_files_properties(Schema.cpp PROPERTIES
  OBJECT_DEPENDS "${tableSchema_HEADER};${utilitySchema_HEADER}"
)

# ---------------------------------------------------------------------------
# Optional: stack frame capture via cpptrace (heavyweight, off by default)
# ---------------------------------------------------------------------------
option(RPD_ENABLE_CPPTRACE "Enable HIP-call stack frame capture via the (heavyweight) cpptrace submodule" OFF)

if(RPD_ENABLE_CPPTRACE)
  set(_cpptrace_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/../cpptrace")
  if(NOT EXISTS "${_cpptrace_src_dir}/CMakeLists.txt")
    message(FATAL_ERROR
      "RPD_ENABLE_CPPTRACE is ON but the cpptrace submodule is not present.\n"
      "Run: git submodule update --init cpptrace")
  endif()
  set(CPPTRACE_BUILD_SHARED ON CACHE BOOL "" FORCE)
  add_subdirectory("${_cpptrace_src_dir}" "${CMAKE_CURRENT_BINARY_DIR}/cpptrace" EXCLUDE_FROM_ALL)
  target_compile_definitions(rpd_tracer_objs PUBLIC RPD_STACKFRAME_SUPPORT)
  target_link_libraries(rpd_tracer_objs PUBLIC cpptrace::cpptrace)
  message(STATUS "rpd_tracer: cpptrace stack frame support ENABLED")
else()
  message(STATUS "rpd_tracer: cpptrace stack frame support disabled (set -DRPD_ENABLE_CPPTRACE=ON to enable)")
endif()

# ---------------------------------------------------------------------------
# ROCm: HIP + whichever GPU activity DataSources are available.
# Every detected backend is compiled in; runtime selects one (see note at
# top of file).
# ---------------------------------------------------------------------------
if(ROCM_ROOT_DIR)
  find_package(hip CONFIG QUIET)

  target_compile_definitions(rpd_tracer_objs PUBLIC __HIP_PLATFORM_AMD__)
  target_include_directories(rpd_tracer_objs PUBLIC
    "${ROCM_ROOT_DIR}/include"
    "${ROCM_ROOT_DIR}/include/hsa"
  )
  if(hip_FOUND)
    target_link_libraries(rpd_tracer_objs PUBLIC hip::host)
  else()
    find_library(AMDHIP64_LIBRARY NAMES amdhip64 HINTS "${ROCM_ROOT_DIR}/lib")
    if(AMDHIP64_LIBRARY)
      target_link_libraries(rpd_tracer_objs PUBLIC "${AMDHIP64_LIBRARY}")
    endif()
  endif()

  set(_rpd_have_gpu_backend OFF)

  # rocprofiler-sdk (skip on ROCm 6.x — see historical Makefile comment: disabled
  # there for performance reasons; only actually wired up for ROCm != 6.x).
  find_package(rocprofiler-sdk CONFIG QUIET)
  set(_rocm_version "")
  if(EXISTS "${ROCM_ROOT_DIR}/.info/version")
    file(READ "${ROCM_ROOT_DIR}/.info/version" _rocm_version)
    string(STRIP "${_rocm_version}" _rocm_version)
  endif()
  string(REGEX MATCH "^[0-9]+" _rocm_major_version "${_rocm_version}")
  if(rocprofiler-sdk_FOUND AND NOT _rocm_major_version STREQUAL "6")
    message(STATUS "rpd_tracer: building with rocprofiler-sdk")
    find_package(nlohmann_json CONFIG REQUIRED)
    target_sources(rpd_tracer_objs PRIVATE RocprofDataSource.cpp)
    target_link_libraries(rpd_tracer_objs PUBLIC
      rocprofiler-sdk::rocprofiler-sdk-shared-library
      nlohmann_json::nlohmann_json
    )
    set(_rpd_have_gpu_backend ON)
  endif()

  # roctracer
  find_path(ROCTRACER_INCLUDE_DIR roctracer/roctracer.h HINTS "${ROCM_ROOT_DIR}/include")
  find_library(ROCTRACER_LIBRARY NAMES roctracer64 HINTS "${ROCM_ROOT_DIR}/lib")
  if(ROCTRACER_INCLUDE_DIR AND ROCTRACER_LIBRARY)
    message(STATUS "rpd_tracer: building with roctracer")
    target_sources(rpd_tracer_objs PRIVATE RoctracerDataSource.cpp)
    target_link_libraries(rpd_tracer_objs PUBLIC "${ROCTRACER_LIBRARY}")
    set(_rpd_have_gpu_backend ON)
  endif()

  # ClrDataSource (CLR profiling extensions, embedded in the HIP runtime)
  if(EXISTS "${ROCM_ROOT_DIR}/include/hip/amd_detail/hip_profiler_ext.h")
    message(STATUS "rpd_tracer: building with ClrDataSource")
    target_sources(rpd_tracer_objs PRIVATE ClrDataSource.cpp)
    set(_rpd_have_gpu_backend ON)
  endif()

  if(NOT _rpd_have_gpu_backend)
    message(STATUS "rpd_tracer: no ROCm GPU activity backend found (rocprofiler-sdk/roctracer/Clr); building without GPU tracing support")
  endif()

  # rocm-trace-lite (RtlDataSource) — optional heavyweight submodule, same
  # opt-in pattern as cpptrace.
  option(RPD_ENABLE_ROCM_TRACE_LITE "Enable RtlDataSource via the (heavyweight) rocm-trace-lite submodule" OFF)
  if(RPD_ENABLE_ROCM_TRACE_LITE)
    set(_rtl_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/../rocm-trace-lite/src")
    if(NOT EXISTS "${_rtl_src_dir}/trace_db.h")
      message(FATAL_ERROR
        "RPD_ENABLE_ROCM_TRACE_LITE is ON but the rocm-trace-lite submodule is not present.\n"
        "Run: git submodule update --init rocm-trace-lite")
    endif()
    find_package(hsa-runtime64 CONFIG REQUIRED)
    message(STATUS "rpd_tracer: building with rocm-trace-lite (RtlDataSource)")
    target_sources(rpd_tracer_objs PRIVATE
      RtlDataSource.cpp
      "${_rtl_src_dir}/hsa_intercept.cpp"
      "${_rtl_src_dir}/hip_api_intercept.cpp"
      "${_rtl_src_dir}/trace_db.cpp"
    )
    target_include_directories(rpd_tracer_objs PRIVATE "${_rtl_src_dir}")
    target_link_libraries(rpd_tracer_objs PUBLIC hsa-runtime64::hsa-runtime64)
  else()
    message(STATUS "rpd_tracer: rocm-trace-lite (RtlDataSource) disabled (set -DRPD_ENABLE_ROCM_TRACE_LITE=ON to enable)")
  endif()
else()
  message(STATUS "rpd_tracer: ROCm not found; building without HIP/GPU tracing support")
endif()

# ---------------------------------------------------------------------------
# Optional: NVIDIA CUPTI backend
# ---------------------------------------------------------------------------
find_path(CUDA_ROOT_DIR NAMES include/cupti.h include/cuda.h
  HINTS "$ENV{CUDA_PATH}" "/usr/local/cuda")

if(CUDA_ROOT_DIR)
  find_library(CUPTI_LIBRARY NAMES cupti HINTS "${CUDA_ROOT_DIR}/lib64" "${CUDA_ROOT_DIR}/extras/CUPTI/lib64")
  if(CUPTI_LIBRARY)
    message(STATUS "rpd_tracer: building with CUPTI (${CUDA_ROOT_DIR})")
    target_sources(rpd_tracer_objs PRIVATE CuptiDataSource.cpp)
    target_include_directories(rpd_tracer_objs PRIVATE "${CUDA_ROOT_DIR}/include")
    target_link_libraries(rpd_tracer_objs PUBLIC "${CUPTI_LIBRARY}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Two shared libraries built from the same objects: librpd_tracer.so
# (LD_PRELOAD, auto-init via constructor/atexit interception) and
# librpd_embedded.so (direct link / dlopen, manual rpdstart()/rpdstop()).
# They differ only in how rpd_capi.cpp is compiled (RPD_TRACER_BUILD).
# ---------------------------------------------------------------------------
add_library(rpd_tracer SHARED rpd_capi.cpp)
set_target_properties(rpd_tracer PROPERTIES
  OUTPUT_NAME "rpd_tracer"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_compile_definitions(rpd_tracer PRIVATE AMD_INTERNAL_BUILD RPD_TRACER_BUILD)
target_link_libraries(rpd_tracer PRIVATE rpd_tracer_objs)
target_link_options(rpd_tracer PRIVATE
  -rdynamic
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/marker_shim.ver"
)

add_library(rpd_embedded SHARED rpd_capi.cpp)
set_target_properties(rpd_embedded PROPERTIES
  OUTPUT_NAME "rpd_embedded"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_compile_definitions(rpd_embedded PRIVATE AMD_INTERNAL_BUILD)
target_link_libraries(rpd_embedded PRIVATE rpd_tracer_objs)
target_link_options(rpd_embedded PRIVATE
  -rdynamic
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/marker_shim.ver"
)

# ---------------------------------------------------------------------------
# rpdrun: plain-C delegate launcher for distributed profiling (rpdrun.c)
# ---------------------------------------------------------------------------
add_executable(rpdrun rpdrun.c)
target_link_libraries(rpdrun PRIVATE ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Tests (rpd_tracer/tests) — Firefly clock-sync convergence / seqlock tests.
# ---------------------------------------------------------------------------
option(RPD_BUILD_TESTS "Build rpd_tracer/tests (Firefly clock-sync unit tests)" OFF)
if(RPD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS rpd_tracer rpd_embedded
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(PROGRAMS runTracer.sh loadTracer.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS rpdrun
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(CODE "execute_process(COMMAND ldconfig)")

find_program(PIP_EXECUTABLE NAMES pip3 pip)
install(CODE "execute_process(
  COMMAND ${PIP_EXECUTABLE} install .
  WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\"
  COMMAND_ERROR_IS_FATAL ANY
)")
