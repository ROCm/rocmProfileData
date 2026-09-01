cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(rlog VERSION 1.0 LANGUAGES CXX C)

find_package(SQLite3 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#configure_file()

add_library(rlog_hub SHARED)
set_target_properties(rlog_hub PROPERTIES OUTPUT_NAME "rlog")

target_include_directories(rlog_hub PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SQLite3_INCLUDE_DIRS}>)

target_sources(rlog_hub PRIVATE
  src/Hub.cpp
  src/PropertyDb.cpp
)

target_link_libraries(rlog_hub PRIVATE ${SQLite3_LIBRARIES})

set_target_properties(rlog_hub PROPERTIES PUBLIC_HEADER "include/rlog/client.h;include/rlog/Logger.h;include/rlog/Range.h;src/Hub.h")

add_executable(rlog_config src/rlog-config.cpp)
set_target_properties(rlog_config PROPERTIES OUTPUT_NAME "rlog-config")
target_link_libraries(rlog_config PRIVATE ${SQLite3_LIBRARIES})

install(TARGETS rlog_config RUNTIME DESTINATION /usr/local/bin)

install (TARGETS rlog_hub
  LIBRARY DESTINATION /usr/local/lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  PUBLIC_HEADER DESTINATION /usr/local/include/rlog
)

install(FILES python/rlog.py
  DESTINATION /usr/local/lib/python/rlog
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

# Mock roctx shared library used by tests
add_library(mock_roctx SHARED tests/mock_roctx.cpp)
set_target_properties(mock_roctx PROPERTIES
  PREFIX ""
  OUTPUT_NAME "mock_roctx")

# test_env: tests RLOG_FORCE_ROCTX and RLOG_ROCTX_LIBPATH
add_executable(test_env tests/test_env.cpp src/rlog.cpp)
target_include_directories(test_env PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(test_env PRIVATE
  MOCK_ROCTX_PATH="$<TARGET_FILE:mock_roctx>")
target_link_libraries(test_env PRIVATE dl)
add_dependencies(test_env mock_roctx)

add_test(NAME env_vars COMMAND test_env)

# test_libpath_no_force: verifies roctx is not enabled without RLOG_FORCE_ROCTX
add_executable(test_libpath_no_force tests/test_libpath_no_force.cpp src/rlog.cpp)
target_include_directories(test_libpath_no_force PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(test_libpath_no_force PRIVATE
  MOCK_ROCTX_PATH="$<TARGET_FILE:mock_roctx>")
target_link_libraries(test_libpath_no_force PRIVATE dl)
add_dependencies(test_libpath_no_force mock_roctx)

add_test(NAME libpath_no_force COMMAND test_libpath_no_force)

# test_roctx_real: tests against real libroctx64.so; skips if not installed
add_executable(test_roctx_real tests/test_roctx_real.cpp src/rlog.cpp)
target_include_directories(test_roctx_real PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_roctx_real PRIVATE dl)

add_test(NAME roctx_real COMMAND test_roctx_real)
set_tests_properties(roctx_real PROPERTIES SKIP_RETURN_CODE 77)

# test_nvtx_real: tests against real libcupti.so; skips if not installed
add_executable(test_nvtx_real tests/test_nvtx_real.cpp src/rlog.cpp)
target_include_directories(test_nvtx_real PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_nvtx_real PRIVATE dl)

add_test(NAME nvtx_real COMMAND test_nvtx_real)
set_tests_properties(nvtx_real PROPERTIES SKIP_RETURN_CODE 77)

# ---------------------------------------------------------------------------
# Benchmarks (standalone, not registered with ctest)
# ---------------------------------------------------------------------------

# bench_loopback: registers a no-op Logger to measure internal dispatch overhead
add_executable(bench_loopback benchmarks/bench_loopback.cpp src/rlog.cpp)
target_include_directories(bench_loopback PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(bench_loopback PRIVATE dl rlog_hub)

# bench_external: no Logger registered; external tool attaches at launch
add_executable(bench_external benchmarks/bench_external.cpp src/rlog.cpp)
target_include_directories(bench_external PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_external PRIVATE dl)

# bench_guard: guards calls with a cached isLogging bool updated via callback
add_executable(bench_guard benchmarks/bench_guard.cpp src/rlog.cpp)
target_include_directories(bench_guard PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_guard PRIVATE dl)

# bench_roctx: same as bench_external but dispatches via roctx; rlog disabled
add_executable(bench_roctx benchmarks/bench_roctx.cpp src/rlog.cpp)
target_include_directories(bench_roctx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_roctx PRIVATE dl)
