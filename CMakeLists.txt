cmake_minimum_required(VERSION 2.8)
project(Demo)
find_package(OpenCV REQUIRED)

#add_subdirectory(xxx)
#include_directories(header)
aux_source_directory(. DIR_SRCS)
add_executable(Camera ${DIR_SRCS})
target_link_libraries(Camera ${OpenCV_LIBS})
