cmake_minimum_required(VERSION 3.16)

project(MovingObjectDetectionCV VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(OpenCV_DIR "/usr/local/opt/opencv/lib/cmake/opencv4")

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

set(SOURCES
    main.cpp
    src/MovingObjectDetector.cpp
)

set(HEADERS
    src/MovingObjectDetector.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

link_directories(${OpenCV_INSTALL_PATH}/lib)
message("These are my opencv libs....: '${OpenCV_INSTALL_PATH}'" )

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

install(TARGETS ${PROJECT_NAME} DESTINATION bin)