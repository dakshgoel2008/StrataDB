cmake_minimum_required(VERSION 3.25)

project(StrataDB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # helps VS Code understand the project

# Build optimized code unless asked otherwise.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(STRATA_BUILD_TESTS "Build unit tests" ON)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
set(STRATA_SANITIZE "" CACHE STRING "Bug-finding checks to enable, e.g. 'address,undefined' or 'thread'")

# Bug-finding checks must cover every file, including the test library.
if(STRATA_SANITIZE)
  add_compile_options(-fsanitize=${STRATA_SANITIZE} -fno-sanitize-recover=all
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${STRATA_SANITIZE})
endif()

# Strict warnings, applied only to our own code.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  -Wall -Wextra -Wpedantic
  -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE -Werror)
endif()

add_subdirectory(src)    # the database engine
add_subdirectory(bench)  # dataset reading and measurement helpers
add_subdirectory(tools)  # small command-line programs

if(STRATA_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  # Download GoogleTest and check the file hasn't been changed.
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    SYSTEM)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()
