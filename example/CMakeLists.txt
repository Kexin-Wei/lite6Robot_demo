CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

FILE(GLOB xarm_app "*.cc")
FOREACH(app ${xarm_app})
  GET_FILENAME_COMPONENT(appname ${app} NAME_WE)
  ADD_EXECUTABLE(${appname} ${app})
  target_include_directories(${appname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lite6RobotxArmCxxSDK/include
  )
  target_link_libraries(${appname} PUBLIC
    debug ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lite6RobotxArmCxxSDK/lib/Debug/xarm.lib
    optimized ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lite6RobotxArmCxxSDK/lib/Release/xarm.lib
  )
  SET_TARGET_PROPERTIES(${appname}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../build_examples)
ENDFOREACH(app)