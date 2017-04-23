cmake_minimum_required(VERSION 2.8)
project(haze_removal)

include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(OpenCV REQUIRED)

add_compile_options(-std=c++11)

# add subdirecories
add_subdirectory(src)
add_subdirectory(test)
