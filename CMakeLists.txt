cmake_minimum_required(VERSION 3.16)

# Project name and version
project(MovingObjectDetectionCV VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


# Tell CMake where to find Qt packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/Qt/6.9.3/macos")

# Tell CMake exactly where OpenCV’s CMakeConfig is
set(OpenCV_DIR "/usr/local/opt/opencv/lib/cmake/opencv4")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Print OpenCV information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

# Source files
set(SOURCES
    main.cpp
    src/MovingObjectDetector.cpp
)

# Header files (for IDE organization)
set(HEADERS
    src/MovingObjectDetector.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#add path to libraries
link_directories(${OpenCV_INSTALL_PATH}/lib)
message("These are my opencv libs....: '${OpenCV_INSTALL_PATH}'" )

# Link libraries - Use target_include_directories and target_link_libraries for better CMake practices
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Print OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 
# Installation rules (optional)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)