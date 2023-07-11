cmake_minimum_required(VERSION 3.10)

FILE(GLOB SRC_FILES "src/*.cpp")

add_library(testLib ${SRC_FILES})

target_include_directories(testLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(testLib PUBLIC
    xarm
)