SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE(TEST main.cpp)

TARGET_LINK_LIBRARIES(TEST ${OpenCV_LIBS} libutils.so)
