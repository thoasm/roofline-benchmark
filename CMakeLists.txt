cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(roofline_benchmark LANGUAGES CXX)

option(ROOFLINE_CUDA "Build the CUDA benchmark" ON)
option(ROOFLINE_HIP "Build the HIP benchmark" OFF)
option(ROOFLINE_CPU "Build the CPU benchmark" OFF)


set(GINKGO_DIR "$ENV{HOME}/projects/ginkgo_github" CACHE PATH
    "Directory of Ginkgo")

function(roofline_apply_default_target_settings target)
    target_compile_features("${target}" PUBLIC cxx_std_14)
    target_include_directories("${target}" PRIVATE
        "${GINKGO_DIR}"
        )
endfunction()


if (ROOFLINE_CPU)
    set(target "roofline_cpu")
    add_executable("${target}")
    roofline_apply_default_target_settings("${target}")
    target_compile_definitions("${target}" PRIVATE ROOFLINE_CPU_CODE)

    target_sources("${target}" PRIVATE
        cpu/main.cpp
        )

    target_compile_options("${target}" PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:--expt-relaxed-constexpr>
        )
    target_compile_options("${target}" PRIVATE -Wall -Wextra -pedantic -Werror)
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
    # Modified version of: https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable/CMakeLists.txt
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    set(target "roofline_hip")

    set(MY_SOURCE_FILES "hip/main.hip.cpp")
    set(MY_TARGET_NAME "${target}")
    set(MY_HIPCC_OPTIONS "-std=c++14") # Needs to be added again
    set(MY_HCC_OPTIONS)
    set(MY_NVCC_OPTIONS "--expt-relaxed-constexpr")

    set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES}
        HIPCC_OPTIONS ${MY_HIPCC_OPTIONS}
        HCC_OPTIONS ${MY_HCC_OPTIONS}
        NVCC_OPTIONS ${MY_NVCC_OPTIONS})
    
    target_compile_definitions("${target}" PRIVATE ROOFLINE_HIP_CODE)
    roofline_apply_default_target_settings("${target}")
endif()
