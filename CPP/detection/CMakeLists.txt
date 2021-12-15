cmake_minimum_required(VERSION 3.10)
message(common_subdir_CMAKE_CURRENT_BINARY_DIR:${CMAKE_CURRENT_BINARY_DIR})

set(OpenCV_DIR /home/zhanggong/lib/opencv4.5/usr/local/lib/cmake/opencv4)
#set(OpenCV_DIR /home/zhanggong/lib/opencv_gpu/usr/local/lib/cmake/opencv4)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
find_package(OpenCV REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}/detection/include
        ${OpenCV_INCLUDE_DIRS}
)
aux_source_directory(${PROJECT_SOURCE_DIR}/detection/src DETECT_SOURCE)
add_library(detect SHARED ${DETECT_SOURCE})

target_link_libraries(detect PUBLIC ${OpenCV_LIBS})

