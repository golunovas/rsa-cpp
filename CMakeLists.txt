cmake_minimum_required(VERSION 3.1)
project(rsa-cpp)

set (CMAKE_CXX_STANDARD 11)

find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS})

add_executable(rsa-cpp rsa_face_detector.cpp main.cpp)
target_link_libraries(rsa-cpp ${Caffe_LIBRARIES} ${OpenCV_LIBS} )