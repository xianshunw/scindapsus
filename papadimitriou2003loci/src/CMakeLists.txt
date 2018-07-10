aux_source_directory(. LIB_SRC)

add_library(loci SHARED ${LIB_SRC})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(loci PROPERTIES OUTPUT_NAME "loci")

target_link_libraries(loci ${OpenCV_LIBS})