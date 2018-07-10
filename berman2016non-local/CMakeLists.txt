cmake_minimum_required(VERSION 3.2)
project(haze_removal)

include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(OpenCV REQUIRED)
find_package(GSL REQUIRED)

add_compile_options(-std=c++11)

# add subdirecories
add_subdirectory(src)
add_subdirectory(test)