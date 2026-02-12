cmake_minimum_required(VERSION 4.2.3)
project(CTorch LANGUAGES C Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
  else()
    message(
      FATAL_ERROR
        "VCPKG_ROOT not set. Please set it to your vcpkg installation.")
  endif()
endif()

option(BUILD_TESTING "Enable building tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenBLAS REQUIRED)
find_library(M_LIB m)

file(GLOB_RECURSE CTORCH_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(ctorch STATIC)
target_sources(ctorch PRIVATE ${CTORCH_SOURCES})

target_include_directories(
  ctorch PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)

target_link_libraries(ctorch PUBLIC OpenMP::OpenMP_C OpenBLAS::OpenBLAS)
if(M_LIB)
  target_link_libraries(ctorch PUBLIC ${M_LIB})
endif()

target_compile_options(ctorch PRIVATE -O3 -march=native -ffast-math)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
