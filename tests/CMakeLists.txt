cmake_minimum_required(VERSION 4.2.3)
project(CTorchTests LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CUNIT REQUIRED cunit)

if(NOT CUnit_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(CUNIT_PKG REQUIRED cunit)
endif()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

add_executable(tests ${TEST_SOURCES})

target_link_libraries(tests PRIVATE ctorch)

if(TARGET CUnit::CUnit)
  target_link_libraries(tests PRIVATE CUnit::CUnit)
  target_include_directories(
    tests PRIVATE $<TARGET_PROPERTY:CUnit::CUnit,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  target_include_directories(tests PRIVATE ${CUNIT_PKG_INCLUDE_DIRS})
  target_link_libraries(tests PRIVATE ${CUNIT_PKG_LIBRARIES})
  target_compile_options(tests PRIVATE ${CUNIT_PKG_CFLAGS_OTHER})
endif()
