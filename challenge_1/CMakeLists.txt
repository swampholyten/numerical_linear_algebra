cmake_minimum_required(VERSION 3.10)
project(Challenge1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/external/stb)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/stb/stb_image.h")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${CMAKE_SOURCE_DIR}/external/stb/stb_image.h")
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/stb/stb_image_write.h")
    file(DOWNLOAD 
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
        "${CMAKE_SOURCE_DIR}/external/stb/stb_image_write.h")
endif()

file(COPY ${CMAKE_SOURCE_DIR}/assets 
     DESTINATION ${CMAKE_BINARY_DIR})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE Eigen3::Eigen)

target_include_directories(main PRIVATE 
    ${CMAKE_SOURCE_DIR}/external
)
