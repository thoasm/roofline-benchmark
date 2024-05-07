#ifndef BENCHMARK_HIP_HIP_HPP_
#define BENCHMARK_HIP_HIP_HPP_

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

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <hiprand/hiprand_kernel.h>

#include "../device_kernels.hpp.inc"


template <typename T>
__device__ T get_random(hiprandState_t*)
{
    static_assert(sizeof(T) > 0,
                  "This function is not implemented for the given type T.");
}

template <>
__device__ float get_random(hiprandState_t* state)
{
    return hiprand_normal(state);
}

template <>
__device__ double get_random(hiprandState_t* state)
{
    return hiprand_normal_double(state);
}

template <>
__device__ std::int32_t get_random(hiprandState_t* state)
{
    return hiprand(state);
}

template <>
__device__ std::int16_t get_random(hiprandState_t* state)
{
    return static_cast<std::int16_t>(hiprand(state));
}

template <>
__device__ gko::acc::posit32_2 get_random(hiprandState_t* state)
{
    return gko::acc::posit32_2{get_random<double>(state)};
}

template <>
__device__ gko::acc::posit16_2 get_random(hiprandState_t* state)
{
    return gko::acc::posit16_2{get_random<float>(state)};
}

template <std::int32_t block_size>
__global__ void initialize_random_kernel(std::int32_t num_cstate,
                                         hiprandState_t* cstate, unsigned seed)
{
    const std::int64_t idx = blockIdx.x * block_size + threadIdx.x;
    if (idx < num_cstate) {
        hiprand_init(seed, idx, 0, cstate + idx);
    }
}

template <std::int32_t block_size, typename T>
__global__ void set_data_random_kernel(std::size_t num_elems, T* data_ptr,
                                       const hiprandState_t* cstate)
{
    const std::int64_t start_idx = blockIdx.x * block_size + threadIdx.x;
    const std::int64_t stride = gridDim.x * blockDim.x;

    auto cstate_elm = cstate[start_idx];

    for (auto idx = start_idx; idx < num_elems; idx += stride) {
        data_ptr[idx] = get_random<T>(&cstate_elm);
    }
}

template <typename T>
void set_data(std::size_t num_elems, T* data_ptr, unsigned seed,
              RandomNumberGenerator& rng)
{
    constexpr std::int32_t block_size{64};
    constexpr std::int32_t max_grid_size{100};
    const dim3 block_(block_size);
    const dim3 grid_(
        std::min(ceildiv(static_cast<std::int32_t>(num_elems), block_size),
                 max_grid_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    const int32_t num_cstate = grid_.x * block_.x;
    if (rng.prepare_for(num_cstate)) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(initialize_random_kernel<block_size>), grid_,
            block_, 0, 0, num_cstate, rng.get_memory(), seed);
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(set_data_random_kernel<block_size, T>),
                       grid_, block_, 0, 0, num_elems, data_ptr,
                       rng.get_memory());
}

template <std::int32_t block_size, typename FrszCompressor>
__global__ void set_frsz2_data_random_kernel(FrszCompressor frsz,
                                             const hiprandState_t* cstate)
{
    auto cstate_elm = cstate[blockIdx.x * block_size + threadIdx.x];

    for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < frsz.get_total_elements(); idx += gridDim.x * blockDim.x) {
        frsz.template compress_gpu_function<block_size>(
            idx, get_random<typename FrszCompressor::fp_type>(&cstate_elm));
    }
}

template <typename FrszCompressor>
void set_frsz2_data(std::size_t num_elems, std::uint8_t* data_ptr,
                    unsigned seed, RandomNumberGenerator& rng)
{
    constexpr std::int32_t block_size{FrszCompressor::max_exp_block_size};
    constexpr std::int32_t max_grid_size{50};
    const dim3 block_(block_size);
    const dim3 grid_(
        std::min(ceildiv(static_cast<std::int32_t>(num_elems), block_size),
                 max_grid_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    const int32_t num_cstate = grid_.x * block_.x;
    if (rng.prepare_for(num_cstate)) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(initialize_random_kernel<block_size>), grid_,
            block_, 0, 0, num_cstate, rng.get_memory(), seed);
    }

    FrszCompressor frsz(data_ptr, num_elems);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            set_frsz2_data_random_kernel<block_size, FrszCompressor>),
        grid_, block_, 0, 0, frsz, rng.get_memory());
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
        std::array<gko::acc::size_type, 1> size{
            {static_cast<gko::acc::size_type>(num_elems)}};
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
                benchmark_accessor_3d_kernel<default_block_size,
                                             outer_work_iters, inner_work_iters,
                                             compute_iters, ArType, range>),
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

template <typename FrszCompressor, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
kernel_runtime_info run_benchmark_frsz(
    std::size_t num_elems, std::uint8_t* data_ptr,
    const typename FrszCompressor::fp_type input)
{
    constexpr int blocks_per_tb = 8;
    constexpr int block_size =
        blocks_per_tb * FrszCompressor::max_exp_block_size;
    const dim3 block_(block_size);
    const dim3 grid_(
        ceildiv(num_elems, inner_work_iters * outer_work_iters * block_size));
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    FrszCompressor frsz(data_ptr, num_elems);

    timer t;
    t.start();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_frsz_kernel<block_size, outer_work_iters,
                                              inner_work_iters, compute_iters,
                                              FrszCompressor>),
        grid_, block_, 0, 0, frsz, input, num_elems);
    t.stop();
    auto kernel_info =
        get_frsz_kernel_info<outer_work_iters, inner_work_iters, compute_iters,
                             FrszCompressor>(num_elems);
    return {kernel_info.bytes, kernel_info.comps, t.get_time()};
}

#endif  // BENCHMARK_HIP_HIP_HPP_
