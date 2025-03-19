cmake_minimum_required(VERSION 3.10)

project(CSS535_Project VERSION 1.0)
project(CSS535_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_executable(CSS535_Project
    src/Main.cu
    src/KernelConfig.cpp
        src/GAKernel.cu
        include/TSP.h
        include/TSP.h
        src/TSP.cpp
        src/TSP_individual.cu
        src/TSP_individual.cu
        include/TSP_individual.hpp
)

target_include_directories(CSS535_Project PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(CSS535_Project PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(CSS535_Project PRIVATE CUDA::cudart)
target_link_libraries(CSS535_Project PRIVATE CUDA::curand)