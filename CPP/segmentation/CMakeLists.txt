cmake_minimum_required(VERSION 3.10)
message(common_subdir_CMAKE_CURRENT_BINARY_DIR:${CMAKE_CURRENT_BINARY_DIR})

set(OpenCV_DIR /home/zhanggong/lib/opencv4.5/usr/local/lib/cmake/opencv4)
#set(OpenCV_DIR /home/zhanggong/lib/opencv_gpu/usr/local/lib/cmake/opencv4)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-gpu-1.5.1/include
        ${PROJECT_SOURCE_DIR}/segmentation/include
        /home/zhanggong/TensorRT-7.1.3.4/include
        /usr/local/cuda/include
)
link_directories( ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-gpu-1.5.1/lib /home/zhanggong/TensorRT-7.1.3.4/lib /usr/local/cuda/lib64/)
aux_source_directory(${PROJECT_SOURCE_DIR}/segmentation/src DETECT_SOURCE)
add_library(segment SHARED ${DETECT_SOURCE})

target_link_libraries(segment PUBLIC ${OpenCV_LIBS} onnxruntime  nvinfer cudart)

