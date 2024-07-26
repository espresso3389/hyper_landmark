cmake_minimum_required(VERSION 3.12)
project(HyperLandmark)

set(SOURCES
    src/face_landmark_model.cpp
    src/feature_descriptor.cpp
    src/helper.cpp
    src/hog.c
)

include_directories(include)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
