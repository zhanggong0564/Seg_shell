cmake_minimum_required(VERSION 3.10)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(OpenCV_DIR /home/zhanggong/lib/opencv4.5/usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(Detinclude ${PROJECT_SOURCE_DIR}/segmentation/include)
set(Caminclude ${PROJECT_SOURCE_DIR}/Camera/include)
#set(Det_LIB /home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/seg_src/CPP/lib/libdetect.so) #设置 so绝对路径
#file(GLOB LIBFILE ${PROJECT_SOURCE_DIR}/lib/*.so)
include_directories(${Detinclude})
include_directories(${Caminclude})
include_directories(
        /home/zhanggong/TensorRT-7.1.3.4/include
        /usr/local/cuda/include
#        ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-gpu-1.2.0/include
        ${PROJECT_SOURCE_DIR}/detection/include
        ${PROJECT_SOURCE_DIR}/common/include
)
link_directories( ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-gpu-1.2.0/lib /home/zhanggong/TensorRT-7.1.3.4/lib /usr/local/cuda/lib64)
#aux_source_directory(${PROJECT_SOURCE_DIR}/demo/demo2.cpp  demo)

#add_executable(demo ${demo})
add_executable(demo demo.cpp)
#add_executable(demo demo2.cpp)
#target_link_libraries(demo ${LIBFILE} ${OpenCV_LIBS} )
#target_link_libraries(demo ${OpenCV_LIBS} detect camera)
target_link_libraries(demo ${OpenCV_LIBS}  camera nvinfer cudart postprocess detect segment)