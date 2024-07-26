cmake_minimum_required(VERSION 3.12)
project(HyperLandmarkLibrary)

set(SOURCES
    src/face_landmark_model.cpp
    src/feature_descriptor.cpp
    src/helper.cpp
    src/hog.c
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(include)

find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
