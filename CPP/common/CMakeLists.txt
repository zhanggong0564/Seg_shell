cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR /home/zhanggong/lib/opencv4.5/usr/local/lib/cmake/opencv4)
#set(OpenCV_DIR /home/zhanggong/lib/opencv_gpu/usr/local/lib/cmake/opencv4)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
find_package(OpenCV REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}/common/include
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/detection/include
)
aux_source_directory(${PROJECT_SOURCE_DIR}/common/src Comm_SOURCE)
add_library(postprocess SHARED ${Comm_SOURCE})

target_link_libraries(postprocess PUBLIC ${OpenCV_LIBS})

