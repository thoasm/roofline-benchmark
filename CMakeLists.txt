cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(roofline_benchmark LANGUAGES CXX CUDA)


set(GINKGO_DIR
    "$ENV{HOME}/projects/ginkgo_github"
    #"$ENV{HOME}/projects/ginkgo_accessor"
    )
#string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
#string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
#string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
#string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=[sm_52,compute_52]")
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")

add_executable(roofline_cuda)


# Force the language to be CUDA (works, but the main.cu approach is cleaner)
# set_source_files_properties(main.cpp.inc PROPERTIES LANGUAGE CUDA)
target_sources(roofline_cuda PRIVATE
    main.cu
    )


#TODO maybe add the Ginkgo Architecture Selector in this project
target_compile_options(roofline_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_52>
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_70,sm_72,sm_75,sm_80>
    #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>
    #$<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_80,code=sm_80>
    # Make sure the cpp files are treated as cu files
    #$<$<COMPILE_LANGUAGE:CUDA>:-x cu>
    )

target_include_directories(roofline_cuda PRIVATE
    "${GINKGO_DIR}"
# Include CUDA header directory in cpp files
#"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )

target_compile_features(roofline_cuda PUBLIC cxx_std_14)

# Command to get PTX code:
#nvcc \
#  -I${HOME}/projects/ginkgo_github \
#  -Wno-deprecated-gpu-targets  --expt-relaxed-constexpr -arch=sm_75 \
#  -O3 -DNDEBUG -std=c++14 --ptx ../main.cu -o main.ptx \
#  && cat main.ptx | c++filt -t > main_deman.ptx
