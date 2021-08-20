cmake_minimum_required(VERSION 3.14)

set(OpenCV_DIR /home/zhanggong/lib/opencv4.5/usr/local/lib/cmake/opencv4)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/Camera/include)
include_directories(${PROJECT_SOURCE_DIR}/Detection/include)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(${PROJECT_SOURCE_DIR}/Camera/src camera_src)
add_library(camera SHARED ${camera_src})
target_link_libraries(camera PUBLIC ${OpenCV_LIBS})
