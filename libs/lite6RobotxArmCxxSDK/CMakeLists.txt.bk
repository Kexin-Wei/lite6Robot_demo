cmake_minimum_required(VERSION 3.10)


add_library(xarm SHARED IMPORTED)

target_include_directories(xarm INTERFACE
    include
)

target_link_directories(xarm INTERFACE
    debug ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug/xarm.lib
    optimized ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/xarm.lib)

message(STATUS "xarm include dir: ${CMAKE_CURRENT_SOURCE_DIR}/include")