aux_source_directory(. TEST_SRC)

# add directories of link libraries
link_directories(${PROJECT_SOURCE_DIR}/lib)

# add cvplot
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(Test ${TEST_SRC})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(Test ${OpenCV_LIBS}  libloci.so libcvplot.a)