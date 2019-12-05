# --------------------------------------------------------------------------
#
# Copyright (C) 2018 by the ExWave authors
#
# This file is part of the ExWave library.
#
# The ExWave library is free software; you can use it, redistribute it,
# and/or modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version. The full text of the
# license can be found in the file LICENSE at the top level of the ExWave
# distribution.
#
# --------------------------------------------------------------------------

# Create output directory:
file(MAKE_DIRECTORY output)

# Set the name of the project and target:
SET(TARGET "linearized_euler")

# Declare all source files the target consists of.
SET(TARGET_SRC
    source/${TARGET}.cc
    source/elementwise_cg.cc
    source/input_parameters.cc
    source/linearized_euler_operations.cc
    source/parameters.cc
    source/time_integrators.cc
    source/utilities.cc
    #source/wave_equation_operations.cc
    )
FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.1.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_P4EST = ON
This feature is OFF in your installation but is required for this program."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()


# Set up unit tests
IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(tests)
ENDIF()
