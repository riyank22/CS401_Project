cmake_minimum_required(VERSION 4.0)
project(ImageProcessorBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 61)

# --- Target 1: CUDA Version ---
add_executable(GPU gpu.cu filters.cuh)
target_link_libraries(GPU PRIVATE m) # For sqrtf in Sobel
set_target_properties(GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Target 2: OpenMP CPU Version ---
find_package(OpenMP REQUIRED)
add_executable(OMP omp.cpp)
target_link_libraries(OMP PRIVATE OpenMP::OpenMP_CXX)

# --- Target 3: Single-Threaded CPU Version ---
add_executable(SingleThread cpu_single.cpp)

# --- Target 4: MPI CPU Version ---
find_package(MPI REQUIRED)
# This assumes your MPI source file is named cpu_mpi.cpp
add_executable(MPI mpi.cpp)
target_link_libraries(MPI PRIVATE MPI::MPI_CXX)