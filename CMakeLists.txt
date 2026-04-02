cmake_minimum_required(VERSION 3.15)
project("cppgrad CXX")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cppgrad main.cpp)
target_include_directories(cppgrad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(cppgrad PRIVATE -O2 -Wall -Wextra)