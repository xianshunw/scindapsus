cmake_minimum_required(VERSION 2.8)
project(haze_removal)

include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(OpenCV REQUIRED)

# add subdirecories
add_subdirectory(src)
add_subdirectory(test)
