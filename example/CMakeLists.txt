cmake_minimum_required(VERSION 3.12)
project(HyperLandmark)

# Add the executable
add_executable(HyperLandmark main.cpp)

# Link the library to the executable
target_link_libraries(HyperLandmark PRIVATE HyperLandmarkLibrary)
