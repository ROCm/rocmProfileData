# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rpd_remote SHARED Remote.cpp)
set_target_properties(rpd_remote PROPERTIES
  OUTPUT_NAME "rpd_remote"
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
)
target_link_options(rpd_remote PRIVATE -rdynamic)
target_link_libraries(rpd_remote PRIVATE ${CMAKE_DL_LIBS})

include(GNUInstallDirs)

install(TARGETS rpd_remote
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(PROGRAMS rpdRemote.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME rpdRemote
)
install(CODE "execute_process(COMMAND ldconfig)")
