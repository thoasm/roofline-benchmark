cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(roofline_benchmark LANGUAGES CXX)

option(ROOFLINE_CUDA "Build the CUDA benchmark" ON)
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
    # setting the default flags like CMAKE_{LANG}_FLAGS_{TYPE}
    # the setting is copied from the default CMAKE_CXX_FLAGS_{TYPE}
    # NEEDS TO BE IN FRONT OF find_package(HIP)
    set(HIP_HIPCC_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Flags used by the HIPCC compiler during DEBUG builds")
    set(HIP_HIPCC_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}" CACHE STRING "Flags used by the HIPCC compiler during MINSIZEREL builds")
    set(HIP_HIPCC_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "Flags used by the HIPCC compiler during RELEASE builds")
    set(HIP_HIPCC_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the HIPCC compiler during RELWITHDEBINFO builds")
    

    # Modified version of: https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable/CMakeLists.txt
    if(NOT DEFINED ROCM_PATH)
        if(NOT DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
        else()
            set(ROCM_PATH $ENV{HIP_PATH} CACHE PATH "Path to which ROCM has been installed")
        endif()
    endif()
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    if(NOT DEFINED HIP_CLANG_PATH)
        if(NOT DEFINED ENV{HIP_CLANG_PATH})
            set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        else()
            set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        endif()
    endif()

    set(ROOFLINE_HIP_CLANG_COMPILER_FLAGS "" CACHE STRING "Set the required HIP CLANG compiler flags.")
    list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")
    #list (APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${HIP_PATH}")

    set(GPU_TARGETS "gfx900;gfx906;gfx908" CACHE STRING "GPU targets to compile for")
    find_package(HIP REQUIRED)
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

    set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    #For older HIP versions (< 3.5)
    #hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES}
    #    HIPCC_OPTIONS ${MY_HIPCC_OPTIONS}
    #    HCC_OPTIONS ${MY_HCC_OPTIONS}
    #    NVCC_OPTIONS ${MY_NVCC_OPTIONS})
    hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES}
        HIPCC_OPTIONS ${MY_HIPCC_OPTIONS}
        HCC_OPTIONS ${MY_HCC_OPTIONS}
        CLANG_OPTIONS ${ROOFLINE_HIP_CLANG_COMPILER_FLAGS}
        NVCC_OPTIONS ${MY_NVCC_OPTIONS})
    #set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "")
    #set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")
    #target_link_libraries("${target}" hip::device)
    
    target_compile_definitions("${target}" PRIVATE ROOFLINE_HIP_CODE)
    roofline_apply_default_target_settings("${target}")
endif()
