set(GTEST_REQUIRED_VERSION 1.15.2)
cmake_policy(SET CMP0135 NEW)

# list(APPEND CMAKE_PREFIX_PATH "${IPSOLVER_THIRD_PARTY_DIR}")
find_package(
  GTest
  ${GTEST_REQUIRED_VERSION}
  NO_MODULE
  QUIET
)

if(NOT TARGET GTest::gtest)
  message(STATUS "IPsolver: Did not find GTest ${GTEST_REQUIRED_VERSION} installed, downloading to "
    "${IPSOLVER_THIRD_PARTY_DIR}")

  include(FetchContent)
  set(FETCHCONTENT_BASE_DIR "${IPSOLVER_THIRD_PARTY_DIR}")
  fetchcontent_declare(
    GTest
    URL https://github.com/google/googletest/archive/504ea69cf7e9947be54f808a09b7b08988e84b5f.zip
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  fetchcontent_makeavailable(GTest)
else()
  get_target_property(GTEST_INCLUDE_DIRS
    GTest::gtest
    INTERFACE_INCLUDE_DIRECTORIES
  )
  message(STATUS "IPsolver: Found GTest installed in ${GTEST_INCLUDE_DIRS}")
endif()
