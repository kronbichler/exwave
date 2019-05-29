# --------------------------------------------------------------------------
#
# Copyright (C) 2018 by the ExWave authors
#
# This file is part of the ExWave library.
#
# The ExWave library is free software; you can use it, redistribute it,
# and/or modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.  The full text of the
# license can be found in the file LICENSE at the top level of the ExWave
# distribution.
#
# --------------------------------------------------------------------------

# This is encoded in .prm files through lines of the form
#    '# mpirun: 4'
# The result is returned in a variable _mpi_count
FUNCTION(get_mpi_count _filename)
  FILE(STRINGS ${_filename} _input_lines
       REGEX "mpirun:")
  IF("${_input_lines}" STREQUAL "")
    SET(_mpi_count 1 PARENT_SCOPE)
  ELSE()
    # go over the (possibly multiple) lines with MPI markers and choose the last
    FOREACH(_input_line ${_input_lines})
     SET(_last_line ${_input_line})
    ENDFOREACH()
    STRING(REGEX REPLACE "^ *# *mpirun: *([0-9]+) *$" "\\1"
           _mpi_count ${_last_line})
    SET(_mpi_count "${_mpi_count}" PARENT_SCOPE)
  endif()
ENDFUNCTION()


#time limit of 10 minutes per test
SET_IF_EMPTY(TEST_TIME_LIMIT 600)

############################

ADD_CUSTOM_TARGET(tests)

SET(_n_tests "0")
FILE(GLOB _tests *.prm)
LIST (SORT _tests)
message("Tests: ${_tests}")
FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT(_test ${_test} NAME_WE)

  MATH(EXPR _n_tests "${_n_tests} + 1")

  # Create the output directory and a rule to generate the input file.  This
  # input file is the one found in the source dir, but it has the correct
  # output directory so that we do not need to specify it by hand
  FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output-${_test})
  FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

  # then generate a rule that runs the test executable and normalizes the
  # output files. Before running, delete prior content of the directory to
  # make sure no dead files are left there. we have to take care of not
  # deleting those files that have been placed there on purpose, however,
  # which are all of the .cmp.notime files.
  GET_MPI_COUNT(${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm)

  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
    COMMAND
      for i in ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/* \; do
        if echo \$i | grep -q -v .cmp.notime \; then
          rm -f \$i \;
        fi \;
      done
    COMMAND mpirun -np ${_mpi_count} ${CMAKE_BINARY_DIR}/explicit_wave
            ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm
            > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output.tmp
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output.tmp
               ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm 
            ${CMAKE_BINARY_DIR}/explicit_wave 
	    ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.output ${_testdepends}
    )

  # The final target for this test
  ADD_CUSTOM_TARGET(${_test}.run
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
    )

  ADD_CUSTOM_TARGET(tests.${_test})

  # create the output.notime target, and also create a similar file
  # for the screen output. the test_name.output files are copied
  # directly from a run but the timer information and memory
  # consumption are not reproducible and need to be stripped out. we
  # only keep track of the lines which contain the term 'error'
  # printed during the program execution, which can then be compared
  # to a reference data set.
  #
  # we then do the same thing also for the screen out
  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
    COMMAND
    cat ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.output
        | egrep 'error'
        > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.output
    )

  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
    COMMAND
    cat ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
        | egrep 'error'
        > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
            ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
    )

  # create the target that compares the .notime with the saved file
  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
    COMMAND
        if (diff ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
              ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
            > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff) \; then
          : \;
        else
          mv ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff
             ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
          echo "******* Error during diffing output results for ${_test}/${_test}" \;
          echo "******* Results are stored in ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed" \;
          echo "******* Check: ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}/${_test}" \;
          echo "******* Diffs are:" \;
          cat ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
          false \;
        fi
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
            ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
    )

  # add the target for this output file to the dependencies of this test
  ADD_CUSTOM_TARGET(${_test}.diff
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
    )

  ADD_DEPENDENCIES(tests ${_test}.diff)
  ADD_TEST(NAME ${_test}
    COMMAND
    ${CMAKE_COMMAND}
    -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -DTESTNAME=${_test}
    -DERROR="Test ${_test} failed"
    -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  SET_TESTS_PROPERTIES(${_test} PROPERTIES
    TIMEOUT ${TEST_TIME_LIMIT}
    )
ENDFOREACH()

MESSAGE(STATUS "Added ${_n_tests} tests")
