aux_source_directory(. LIB_SRC)

add_library(tools SHARED ${LIB_SRC})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(tools PROPERTIES OUTPUT_NAME "tools")

target_link_libraries(tools ${OpenCV_LIBS})
