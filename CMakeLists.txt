cmake_minimum_required(VERSION 3.10)
project(CTorch C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE C_TORCH_SOURCES CONFIGURE_DEPENDS "src/*.c")

add_library(ctorch ${C_TORCH_SOURCES})
target_include_directories(ctorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ctorch PUBLIC m)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CUNIT REQUIRED cunit)
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.c")

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests ctorch ${CUNIT_LIBRARIES})

target_include_directories(tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(tests PRIVATE ${CUNIT_INCLUDE_DIRS})
