# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright (c) 2025, Davide Stocco and Enrico Bertolazzi.                                        #                                                              #
#                                                                                                 #
# The IPsolver project is distributed under the MIT License.                                      #
#                                                                                                 #
# Davide Stocco                                                                 Enrico Bertolazzi #
# University of Trento                                                       University of Trento #
# e-mail: davide.stocco@unitn.it                               e-mail: enrico.bertolazzi@unitn.it #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(IPsolverVersion)
get_version_from_git()

project(IPsolver
  VERSION ${IPSOLVER_VERSION}
  DESCRIPTION "A minimal C++ IP solver"
  HOMEPAGE_URL "https://github.com/StoccoDavide/IPsolver"
  LANGUAGES CXX
)

if(NOT IPSOLVER_VERSION_PRERELEASE)
  set(IPSOLVER_FULL_VERSION "${IPSOLVER_VERSION}")
  write_version_file()
else()
  set(IPSOLVER_FULL_VERSION "${IPSOLVER_VERSION}-${IPSOLVER_VERSION_PRERELEASE}")
endif()
message(STATUS "IPsolver: version ${IPSOLVER_FULL_VERSION}")

set(IPSOLVER_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/lib")
  message(STATUS "IPsolver: CMAKE_INSTALL_PREFIX not specified, defaulting to ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Dependencies
include(IPsolverEigen3)

# Library definition
add_library(IPsolver INTERFACE)
add_library(IPsolver::IPsolver ALIAS IPsolver)

target_link_libraries(IPsolver INTERFACE Eigen3::Eigen)

target_include_directories(IPsolver INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# INFO: Ensure this options are only available to developers. In theory we could leave them available,
# but the call to `target_sources` breaks `fetchcontent` compatibility due to absolute paths being
# added to `INTERFACE_SOURCES`. I tried solving it, but it seems to be poorly documented, supported, etc.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    message(STATUS
      "IPsolver: "
      "CMAKE_BUILD_TYPE not specified, defaulting to ${CMAKE_BUILD_TYPE}"
    )
  endif()

  option(IPSOLVER_BUILD_TESTS "Build tests" OFF)
  option(IPSOLVER_BUILD_EXAMPLES "Build examples" OFF)

  file(GLOB_RECURSE HEADER_FILES_HH "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh")
  foreach(HEADER_FILE IN LISTS HEADER_FILES_HH)
    file(
      RELATIVE_PATH
      RELATIVE_PATH_HEADER_FILE
      ${CMAKE_CURRENT_SOURCE_DIR} ${HEADER_FILE}
    )
    list(APPEND RELATIVE_HEADER_FILES ${RELATIVE_PATH_HEADER_FILE})
  endforeach()
  target_sources(IPsolver INTERFACE $<BUILD_INTERFACE:${RELATIVE_HEADER_FILES}>)

  if(IPSOLVER_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endif()

  if(IPSOLVER_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
  endif()

endif()

if(MSVC)
  target_compile_options(IPsolver INTERFACE /W4 /WX)
else()
  target_compile_options(IPsolver INTERFACE -Wall -Wextra -pedantic -Werror) # -Wno-narrowing
endif()

# Installation
install(TARGETS IPsolver EXPORT IPsolverTargets)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)

export(EXPORT IPsolverTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/IPsolverTargets.cmake
  NAMESPACE IPsolver::
)

install(EXPORT IPsolverTargets
  FILE IPsolverTargets.cmake
  NAMESPACE IPsolver::
  DESTINATION lib/cmake/IPsolver
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/IPsolverConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/IPsolverConfig.cmake
  INSTALL_DESTINATION lib/cmake/IPsolver
)

write_basic_package_version_file(
  IPsolverConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/IPsolverConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/IPsolverConfigVersion.cmake
  DESTINATION lib/cmake/IPsolver
)
