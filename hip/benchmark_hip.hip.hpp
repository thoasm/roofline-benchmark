#ifndef BENCHMARK_CUDA_CUH_
#define BENCHMARK_CUDA_CUH_

#include <hip/hip_runtime.h>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>

//
#include <array>
#include <cinttypes>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../benchmark_info.hpp"
#include "../exec_helper.hpp"

//
#include "../device_kernels.hpp.inc"

template <typename T>
void set_data(std::size_t num_elems, T* data_ptr, unsigned seed,
              RandomNumberGenerator& rng)
{
    const dim3 block_(default_block_size);
    const dim3 grid_(ceildiv(num_elems, default_block_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(set_data_kernel<default_block_size, T>),
                       grid_, block_, 0, 0, num_elems, data_ptr, seed);
}


template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
kernel_runtime_info run_benchmark_hand(std::size_t num_elems, T* data_ptr,
                                       const T input)
{
    const dim3 block_(default_block_size);
    const dim3 grid_(ceildiv(
        num_elems, inner_work_iters * outer_work_iters * default_block_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    timer t;
    t.start();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_kernel<default_block_size, outer_work_iters,
                                         inner_work_iters, compute_iters, T>),
        grid_, block_, 0, 0, data_ptr, input);
    t.stop();
    auto kernel_info =
        get_kernel_info<outer_work_iters, inner_work_iters, compute_iters, T>(
            num_elems);
    return {kernel_info.bytes, kernel_info.comps, t.get_time()};
}

template <typename ArType, typename StType, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters,
          std::size_t dimensionality>
kernel_runtime_info run_benchmark_accessor(std::size_t num_elems,
                                           StType* data_ptr, const ArType input)
{
    const dim3 block_(default_block_size);
    const dim3 grid_(ceildiv(
        num_elems, inner_work_iters * outer_work_iters * default_block_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }
    // auto total_threads = static_cast<std::size_t>(block_.x) * block_.y *
    //                     block_.z * grid_.x * grid_.y * grid_.z;
    auto total_threads = static_cast<gko::acc::size_type>(block_.x) * grid_.x;

    //*
    static_assert(dimensionality == 1 || dimensionality == 3,
                  "Dimensionality must be 1 or 3!");
    if (dimensionality == 1) {
        std::array<gko::acc::size_type, 1> size{{num_elems}};
        std::array<gko::acc::size_type, 1 - 1> stride{};
        using accessor = gko::acc::reduced_row_major<1, ArType, StType>;
        using range = gko::acc::range<accessor>;
        auto acc = range(size, data_ptr, stride);

        timer t;
        t.start();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                benchmark_accessor_1d_kernel<default_block_size,
                                             outer_work_iters, inner_work_iters,
                                             compute_iters, ArType, range>),
            grid_, block_, 0, 0, acc, input);
        t.stop();
        auto kernel_info = get_kernel_info<outer_work_iters, inner_work_iters,
                                           compute_iters, StType>(num_elems);
        return {kernel_info.bytes, kernel_info.comps, t.get_time()};
    } else if (dimensionality == 3) {  // dimensionality == 3

        std::array<gko::acc::size_type, 3> size{
            {outer_work_iters, inner_work_iters, total_threads}};
        std::array<gko::acc::size_type, 3 - 1> stride{
            static_cast<gko::acc::size_type>(grid_.x) * block_.x *
                inner_work_iters,
            static_cast<gko::acc::size_type>(block_.x)};
        using accessor = gko::acc::reduced_row_major<3, ArType, StType>;
        using range = gko::acc::range<accessor>;
        auto acc = range(size, data_ptr, stride);

        timer t;
        t.start();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                benchmark_accessor_kernel<default_block_size, outer_work_iters,
                                          inner_work_iters, compute_iters,
                                          ArType, range>),
            grid_, block_, 0, 0, acc, input);
        t.stop();
        auto kernel_info = get_kernel_info<outer_work_iters, inner_work_iters,
                                           compute_iters, StType>(num_elems);
        return {kernel_info.bytes, kernel_info.comps, t.get_time()};
    } else {
        throw std::invalid_argument(
            std::string("Invalid dimensionality value! ") +
            std::to_string(dimensionality) + " is not supported!");
    }
}

#endif  // BENCHMARK_CUDA_CUH_
