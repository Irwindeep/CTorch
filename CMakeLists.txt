cmake_minimum_required(VERSION 4.2.3)
project(CTorch LANGUAGES C Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

option(BUILD_TESTING "Enable building tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenBLAS REQUIRED)
find_library(M_LIB m)

file(GLOB_RECURSE CTORCH_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

# Core library

add_library(ctorch SHARED ${CTORCH_SOURCES})

target_include_directories(
  ctorch PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)

target_link_libraries(ctorch PUBLIC OpenMP::OpenMP_C OpenBLAS::OpenBLAS)

if(M_LIB)
  target_link_libraries(ctorch PUBLIC ${M_LIB})
endif()

# Vision library

find_package(PkgConfig REQUIRED)
pkg_check_modules(VIPS REQUIRED IMPORTED_TARGET vips)

file(GLOB_RECURSE CTORCH_VISION_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/modules/vision/*.c")

add_library(ctorchvision SHARED ${CTORCH_VISION_SOURCES})

target_include_directories(
  ctorchvision PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

target_link_libraries(ctorchvision PUBLIC ctorch OpenMP::OpenMP_C
                                          PkgConfig::VIPS)

# Compile flags

add_library(ctorch_warnings INTERFACE)

target_compile_options(
  ctorch_warnings
  INTERFACE -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wformat=2
            -Wvla
            -Wnull-dereference
            -Wdouble-promotion
            -Wno-unused)

target_link_libraries(ctorch PRIVATE ctorch_warnings)
target_link_libraries(ctorchvision PRIVATE ctorch_warnings)

# Tests

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
