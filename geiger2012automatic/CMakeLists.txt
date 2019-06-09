cmake_minimum_required(VERSION 2.8)

project(MULTICHESSBOARD)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(main main.cpp utils.cpp)

target_link_libraries(main ${OpenCV_LIBS})

