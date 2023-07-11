CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

FILE(GLOB xarm_app "*.cc")
FOREACH(app ${xarm_app})

  GET_FILENAME_COMPONENT(appname ${app} NAME_WE)
  ADD_EXECUTABLE(${appname} ${app})

  target_include_directories(${appname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../lite6SDK/include
  )

  if (WIN32)
    target_link_libraries(${appname} PUBLIC
      debug ${CMAKE_CURRENT_SOURCE_DIR}/../lite6SDK/lib/Debug/xarm.lib
      optimized ${CMAKE_CURRENT_SOURCE_DIR}/../lite6SDK/lib/Release/xarm.lib
    )
  elseif (UNIX)
    target_link_libraries(${appname} PUBLIC
      debug ${CMAKE_CURRENT_SOURCE_DIR}/../lite6SDK/lib/Debug/libxarm.so
      optimized ${CMAKE_CURRENT_SOURCE_DIR}/../lite6SDK/lib/Release/libxarm.so
    )
  endif()

  if(WIN32)
    install(FILES $<TARGET_PDB_FILE:${appname}> DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG> OPTIONAL)
  endif()

  SET_TARGET_PROPERTIES(${appname}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../../build_examples)
ENDFOREACH(app)