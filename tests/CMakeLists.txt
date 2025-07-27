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

file(GLOB_RECURSE TEST_SMALL_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/test_small_example.cc")
add_executable(test_small_example ${TEST_SMALL_EXAMPLE})
target_link_libraries(test_small_example PRIVATE IPsolver Catch2::Catch2WithMain)
