# SPDX-FileCopyrightText: 2023 Kristin Kexin Wei <wkx611@outlook.com>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)

project(lite6RobotDemo)

add_subdirectory(lite6SDK)

add_executable(lite6RobotDemo main.cpp)
add_subdirectory(example)
add_subdirectory(libs/test)

target_link_libraries(lite6RobotDemo PUBLIC
    xarm
    testApi
)

if(WIN32)
    install(FILES $<TARGET_PDB_FILE:lite6RobotDemo> DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG> OPTIONAL)
endif()
