cmake_minimum_required(VERSION 4.0)
project(GPU LANGUAGES CXX CUDA )

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(GPU main.cu
        filters.cuh)

set_target_properties(GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
find_package(OpenMP REQUIRED)
add_executable(OMP cpu.cpp)
target_link_libraries(OMP PRIVATE OpenMP::OpenMP_CXX)

add_executable(ProjectCode_ST cpu_single.cpp)

set_target_properties(GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
