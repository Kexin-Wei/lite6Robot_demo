cmake_minimum_required(VERSION 3.10)

add_library(xarm SHARED IMPORTED GLOBAL)

target_include_directories(xarm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# set lib for windows and linux
if(WIN32)
    set_target_properties(xarm PROPERTIES
        IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug/xarm.dll
        IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/xarm.dll
    )
elseif(UNIX)
    set_target_properties(xarm PROPERTIES
        IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug/libxarm.so
        IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/libxarm.so
    )
endif()

# message(STATUS "xarm headers: ${CMAKE_CURRENT_SOURCE_DIR}/include") 