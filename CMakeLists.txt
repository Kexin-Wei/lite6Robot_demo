# SPDX-FileCopyrightText: 2023 Kristin Kexin Wei <wkx611@outlook.com>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(lite6RobotDemo)

add_subdirectory(lite6SDK)
add_subdirectory(example)
add_subdirectory(libs)

add_executable(lite6RobotDemo main.cpp)


target_link_libraries(lite6RobotDemo PUBLIC
    xarm
    testLib
)

if(WIN32)
    install(FILES $<TARGET_PDB_FILE:lite6RobotDemo> DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG> OPTIONAL)
endif()
