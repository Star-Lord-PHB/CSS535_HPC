cmake_minimum_required(VERSION 3.10)

project(CSS535_Project VERSION 1.0)
project(CSS535_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_executable(CSS535_Project
    src/Main.cu
    src/TSP.cpp
    include/GAInterface.h
    include/GA_CPU.h
    include/GA_CUDA.h
    src/GA_CPU.cpp
    src/GA_CUDA.cu
    include/TSP.h
    src/GA_oneThreadPerGene.cu
    include/Array.hpp
)

#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
#endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")


target_include_directories(CSS535_Project PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(CSS535_Project PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(CSS535_Project PRIVATE CUDA::cudart)
target_link_libraries(CSS535_Project PRIVATE CUDA::curand)