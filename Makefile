# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Thin wrapper around the CMake build, kept for convenience / muscle memory.
# See CMakeLists.txt for the actual build definition and available options
# (e.g. -DRPD_ENABLE_CPPTRACE=ON, -DRPD_ENABLE_ROCM_TRACE_LITE=ON).

BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release
CMAKE_ARGS ?=

.PHONY: all
all:
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(shell nproc)

.PHONY: install
install: all
	cmake --install $(BUILD_DIR)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
