cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(roofline_benchmark LANGUAGES CXX)

option(ROOFLINE_CUDA "Build the CUDA benchmark" OFF)
option(ROOFLINE_HIP "Build the HIP benchmark" OFF)
option(ROOFLINE_CPU "Build the CPU benchmark" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(GINKGO_DIR "$ENV{HOME}/projects/ginkgo_github" CACHE PATH
    "Directory of Ginkgo")

function(roofline_apply_default_target_settings target)
    target_compile_features("${target}" PUBLIC cxx_std_14)
    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
        )
endfunction()


if (ROOFLINE_CPU)
    find_package(OpenMP REQUIRED)
    set(target "roofline_cpu")
    add_executable("${target}")
    target_link_libraries("${target}" PUBLIC OpenMP::OpenMP_CXX)
    roofline_apply_default_target_settings("${target}")
    target_compile_definitions("${target}" PRIVATE ROOFLINE_CPU_CODE)

    target_sources("${target}" PRIVATE
        cpu/main.cpp
        )

    target_compile_options("${target}" PRIVATE 
        -Wall
        -Wextra
        -pedantic
        -march=native
        -Wno-unused-parameter
        )
endif()

if (ROOFLINE_CUDA)
    enable_language(CUDA)
    set(target "roofline_cuda")
    add_executable("${target}")
    roofline_apply_default_target_settings("${target}")
    target_compile_definitions("${target}" PRIVATE ROOFLINE_CUDA_CODE)

    # Force the language to be CUDA (works, but the main.cu approach is cleaner)
    # set_source_files_properties(main.cpp.inc PROPERTIES LANGUAGE CUDA)
    target_sources("${target}" PRIVATE
        cuda/main.cu
        )


    #TODO maybe add the Ginkgo Architecture Selector in this project
    target_compile_options("${target}" PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_52>
        $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_70,sm_72,sm_75,sm_80>
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>
        #$<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_80,code=sm_80>
        # Make sure the cpp files are treated as cu files
        #$<$<COMPILE_LANGUAGE:CUDA>:-x cu>
        )

    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
    # Include CUDA header directory in cpp files
    #"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        )


    # Command to get PTX code:
    #nvcc \
    #  -I${HOME}/projects/ginkgo_github \
    #  -Wno-deprecated-gpu-targets  --expt-relaxed-constexpr -arch=sm_75 \
    #  -O3 -DNDEBUG -std=c++14 --ptx ../main.cu -o main.ptx \
    #  && cat main.ptx | c++filt -t > main_deman.ptx
endif()

if (ROOFLINE_HIP)
    set(GPU_TARGETS "gfx900;gfx906;gfx908" CACHE STRING "GPU targets to compile for")
    enable_language(HIP)
    find_package(HIP REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(rocthrust REQUIRED)
    find_package(rocrand REQUIRED)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    set(target "roofline_hip")

    if(GPU_TARGETS)
        foreach(gpu_target "${GPU_TARGETS}")
            list(APPEND ROOFLINE_HIP_CLANG_COMPILER_FLAGS --amdgpu-target=${gpu_target})
        endforeach()
    endif()
    
    set(MY_SOURCE_FILES "hip/main.hip.cpp")
    set(MY_TARGET_NAME "${target}")
    set(MY_HIPCC_OPTIONS
        "-std=c++14" # Needs to be added again
        "${OPTIONAL_NDEBUG}" # If release, define NDEBUG
        )
    set(MY_HCC_OPTIONS)
    set(MY_NVCC_OPTIONS "--expt-relaxed-constexpr")

    #set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
    add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES})
    target_link_libraries(${MY_TARGET_NAME} PRIVATE hip::host hip::hiprand roc::rocthrust roc::rocrand)
    #set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "")
    #set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")
    #target_link_libraries("${target}" hip::device)
    
    target_compile_definitions("${target}" PRIVATE ROOFLINE_HIP_CODE)
    roofline_apply_default_target_settings("${target}")
endif()
