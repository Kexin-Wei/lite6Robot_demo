cmake_minimum_required(VERSION 3.10)

FILE(GLOB_RECURSE SRC_FILES "*.cpp")
FILE(GLOB_RECURSE HEADER_FILES "*.h")

add_library(testApi ${SRC_FILES} ${HEADER_FILES})

target_include_directories(testApi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lite6SDK/include
)
if (WIN32)
    target_link_libraries(testApi PUBLIC
        debug ${CMAKE_CURRENT_SOURCE_DIR}/../../lite6SDK/lib/Debug/xarm.lib
        optimized ${CMAKE_CURRENT_SOURCE_DIR}/../../lite6SDK/lib/Release/xarm.lib
    )
elseif (UNIX)
    target_link_libraries(testApi PUBLIC
        debug ${CMAKE_CURRENT_SOURCE_DIR}/../../lite6SDK/lib/Debug/libxarm.so
        optimized ${CMAKE_CURRENT_SOURCE_DIR}/../../lite6SDK/lib/Release/libxarm.so
    )
endif()

message(STATUS "testApi header files: ${HEADER_FILES}")