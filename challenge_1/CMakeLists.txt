cmake_minimum_required(VERSION 3.10)
project(Challenge_1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(COPY ${CMAKE_SOURCE_DIR}/assets 
     DESTINATION ${CMAKE_BINARY_DIR})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main
    src/main.cpp
)

target_link_libraries(main Eigen3::Eigen)
