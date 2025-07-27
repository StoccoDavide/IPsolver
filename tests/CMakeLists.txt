# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                        #
#                                                                                                 #
# The IPsolver project is distributed under the MIT License.                                      #
#                                                                                                 #
# Davide Stocco                                                                 Enrico Bertolazzi #
# University of Trento                                                       University of Trento #
# e-mail: davide.stocco@unitn.it                               e-mail: enrico.bertolazzi@unitn.it #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

include(IPsolverCatch2)

file(GLOB_RECURSE TEST_QPROG "${CMAKE_CURRENT_SOURCE_DIR}/test_qprog.cc")
add_executable(test_qprog ${TEST_QPROG})
target_link_libraries(test_qprog PRIVATE IPsolver Catch2::Catch2WithMain)

file(GLOB_RECURSE TEST_LOGISTIC "${CMAKE_CURRENT_SOURCE_DIR}/test_logistic.cc")
add_executable(test_logistic ${TEST_LOGISTIC})
target_link_libraries(test_logistic PRIVATE IPsolver Catch2::Catch2WithMain)
