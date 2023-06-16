cmake_minimum_required(VERSION 3.10)

add_library(xarm SHARED IMPORTED)

target_include_directories(xarm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(xarm PROPERTIES
    IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug/xarm.lib
    IMPORTED_IMPLIB_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/xarm.lib
)

message(STATUS "xarm headers: ${CMAKE_CURRENT_SOURCE_DIR}/include") 