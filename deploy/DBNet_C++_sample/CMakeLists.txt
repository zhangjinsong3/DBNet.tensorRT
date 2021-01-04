cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(DBNet_C++)

MESSAGE(STATUS "gcc: ${CMAKE_C_COMPILER}, g++: ${CMAKE_CXX_COMPILER}")

ADD_DEFINITIONS(-DDEBUG)  # comment if u do not want to see debug infomation

include_directories(
        utils
)
# tensorRT 6.0.1.5
set(TensorRT_DIR /home/zjs/opt/TensorRT-6.0.1.5)

# Opencv 3.4.0
set(OpenCV_DIR /home/zjs/opt/OpenCV-3.4.0-cuda-9.0/share/OpenCV)
find_package(OpenCV REQUIRED)

# cuda
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${TensorRT_DIR}/include /usr/local/cuda/include)
link_directories(${TensorRT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)     #  -L
add_executable(DBNet_C++ main.cpp db_detector.cpp db_detector.h utils/filesystem.cpp utils/clipper/clipper.cpp)

target_link_libraries(DBNet_C++ utils
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES} cublas cudnn nvinfer nvparsers nvonnxparser nvonnxparser_runtime nvinfer_plugin)

add_subdirectory(utils)
add_subdirectory(utils/simpleini)

set_property(TARGET DBNet_C++ PROPERTY CXX_STANDARD 14)
