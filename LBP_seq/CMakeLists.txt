cmake_minimum_required(VERSION 3.28)
project( LocalBinaryPattern_seq )

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( LocalBinaryPattern_seq main.cpp
                                        LBP.cpp
                                        LBP.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(LocalBinaryPattern_seq ${OpenCV_LIBS})