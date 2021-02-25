#ifndef BENCHMARK_CUDA_CUH_
#define BENCHMARK_CUDA_CUH_

#include <hip/hip_runtime.h>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>

//
#include <array>
#include <cinttypes>
#include <iostream>
#include <type_traits>

#include "benchmark_info.hpp"
#include "exec_helper.hpp"

//
#include "device_kernels.hpp.inc"

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
void run_benchmark_hand(std::size_t num_elems, T input, T *data_ptr) {
    const dim3 block_(block_size);
    const dim3 grid_(
        ceildiv(num_elems, inner_work_iters * outer_work_iters * block_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_kernel<block_size, outer_work_iters,
                                         inner_work_iters, compute_iters, T>),
        grid_, block_, 0, 0, input, data_ptr);
}

template <typename ArType, typename StType, std::int32_t block_size,
          std::int32_t outer_work_iters, std::int32_t inner_work_iters,
          std::int32_t compute_iters, std::size_t dimensionality>
void run_benchmark_accessor(std::size_t num_elems, ArType input,
                            StType *data_ptr) {
    const dim3 block_(block_size);
    const dim3 grid_(
        ceildiv(num_elems, inner_work_iters * outer_work_iters * block_size));
    auto total_threads = static_cast<std::size_t>(block_.x) * block_.y *
                         block_.z * grid_.x * grid_.y * grid_.z;
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    //*
    static_assert(dimensionality == 3, "Dimensionality must be 3!");
    std::array<std::size_t, dimensionality> size{
        {outer_work_iters, inner_work_iters, total_threads}};
    /*/
    static_assert(dimensionality == 1, "Dimensionality must be 1!");
    std::array<std::size_t, dimensionality> size{{outer_work_iters *
    inner_work_iters * total_threads}};
    //*/
    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    auto acc = range(size, data_ptr);
    // Warmup
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            benchmark_accessor_kernel<block_size, outer_work_iters,
                                      inner_work_iters, compute_iters, ArType,
                                      range>),
        grid_, block_, 0, 0, input, acc);
}

#endif  // BENCHMARK_CUDA_CUH_
