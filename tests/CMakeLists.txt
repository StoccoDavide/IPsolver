# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                        #
#                                                                                                 #
# The IPsolver project is distributed under the MIT License.                                      #
#                                                                                                 #
# Davide Stocco                                                                 Enrico Bertolazzi #
# University of Trento                                                       University of Trento #
# e-mail: davide.stocco@unitn.it                               e-mail: enrico.bertolazzi@unitn.it #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

include(IPsolverGTest)

file(GLOB_RECURSE TEST_LINEAR "${CMAKE_CURRENT_SOURCE_DIR}/test_linear.cc")
add_executable(test_linear ${TEST_LINEAR})
target_link_libraries(test_linear PRIVATE IPsolver GTest::gtest_main)

file(GLOB_RECURSE TEST_QUADRATIC "${CMAKE_CURRENT_SOURCE_DIR}/test_quadratic.cc")
add_executable(test_quadratic ${TEST_QUADRATIC})
target_link_libraries(test_quadratic PRIVATE IPsolver GTest::gtest_main)

file(GLOB_RECURSE TEST_LOGISTIC "${CMAKE_CURRENT_SOURCE_DIR}/test_logistic.cc")
add_executable(test_logistic ${TEST_LOGISTIC})
target_link_libraries(test_logistic PRIVATE IPsolver GTest::gtest_main)
