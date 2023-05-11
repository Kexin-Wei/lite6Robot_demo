cmake_minimum_required(VERSION 3.10)

project(lite6RobotDemo)

add_subdirectory(libs/lite6RobotxArmCxxSDK)

add_executable(lite6RobotDemo main.cpp)

target_link_libraries(lite6RobotDemo PUBLIC
	xarm
)