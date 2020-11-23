cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(roofline_benchmark LANGUAGES CXX CUDA)

# Include CUDA header directory in cpp files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
#string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=[sm_52,compute_52] -gencode arch=compute_75,code=compute_75")

add_executable(roofline_cuda)

target_sources(roofline_cuda PRIVATE
    main.cu
    )
target_include_directories(roofline_cuda PRIVATE
    "$ENV{HOME}/projects/ginkgo_github/include"
    "$ENV{HOME}/projects/ginkgo_github/build/debug/include"
    "$ENV{HOME}/projects/ginkgo_github"
    )

target_compile_features(roofline_cuda PUBLIC cxx_std_14)

# Command to get PTX code:
#nvcc   -I/home/thomas/projects/ginkgo_github/include -I/home/thomas/projects/ginkgo_github/build/debug/include -I/home/thomas/projects/ginkgo_github -Wno-deprecated-gpu-targets -arch=sm_75 -O3 -DNDEBUG -std=c++14 --ptx ../main.cu -o main.ptx && cat main.ptx | c++filt -t > main_deman.ptx
