# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Firefly clock-sync unit tests (seqlock / measurement / regression /
# convergence). These only need Firefly.cpp + Utility.cpp from the parent
# directory, mirroring rpd_tracer/tests/Makefile's PARENT_OBJS.

set(RPD_TEST_PARENT_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../Firefly.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../Utility.cpp
)

set(RPD_TESTS
  test_seqlock
  test_measurement
  test_regression
  test_convergence
)

foreach(_test ${RPD_TESTS})
  add_executable(${_test} ${_test}.cpp ${RPD_TEST_PARENT_SRCS})
  target_include_directories(${_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set_target_properties(${_test} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
  )
  target_link_libraries(${_test} PRIVATE Threads::Threads)
  add_test(NAME ${_test} COMMAND ${_test})
endforeach()

# bench_rlog_strings: standalone benchmark, not registered with ctest (needs
# a sample .rpd file and is meant to be run under runTracer.sh).
find_package(SQLite3 REQUIRED)
add_executable(bench_rlog_strings EXCLUDE_FROM_ALL bench_rlog_strings.cpp)
target_include_directories(bench_rlog_strings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../rlog/include
)
set_target_properties(bench_rlog_strings PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
)
target_link_libraries(bench_rlog_strings PRIVATE SQLite::SQLite3 rlog_hub)
