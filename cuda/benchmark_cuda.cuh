#ifndef BENCHMARK_CUDA_CUH_
#define BENCHMARK_CUDA_CUH_

#include <accessor/posit.hpp>
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>


#include <array>
#include <cinttypes>
#include <iostream>
#include <type_traits>


#include <curand_kernel.h>


#include "../benchmark_info.hpp"
#include "../exec_helper.hpp"


#include "../device_kernels.hpp.inc"


template <typename T>
T get_random(curandState_t*)
{
    static_assert(sizeof(T) > 0,
                  "This function is not implemented for the given type T.");
}

template <>
__device__ float get_random(curandState_t* state)
{
    return curand_normal(state);
}

template <>
__device__ double get_random(curandState_t* state)
{
    return curand_normal_double(state);
}

template <>
__device__ std::int32_t get_random(curandState_t* state)
{
    return curand(state);
}

template <>
__device__ std::int16_t get_random(curandState_t* state)
{
    return static_cast<std::int16_t>(curand(state));
}

template <>
__device__ gko::acc::posit32_2 get_random(curandState_t* state)
{
    return gko::acc::posit32_2{get_random<double>(state)};
}

template <>
__device__ gko::acc::posit16_2 get_random(curandState_t* state)
{
    return gko::acc::posit16_2{get_random<float>(state)};
}


template <std::int32_t block_size>
__global__ void initialize_random_kernel(std::int32_t num_cstate,
                                         curandState_t* cstate, unsigned seed)
{
    const std::int64_t idx = blockIdx.x * block_size + threadIdx.x;
    if (idx < num_cstate) {
        curand_init(seed, idx, 0, cstate + idx);
    }
}


template <std::int32_t block_size, typename T>
__global__ void set_data_random_kernel(std::size_t num_elems, T* data_ptr,
                                       const curandState_t* cstate)
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
        initialize_random_kernel<block_size>
            <<<grid_, block_>>>(num_cstate, rng.get_memory(), seed);
    }

    set_data_random_kernel<block_size, T>
        <<<grid_, block_>>>(num_elems, data_ptr, rng.get_memory());
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
    benchmark_kernel<default_block_size, outer_work_iters, inner_work_iters,
                     compute_iters, T><<<grid_, block_>>>(data_ptr, input);
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
    static_assert(dimensionality == 3, "Dimensionality must be 3!");
    std::array<gko::acc::size_type, dimensionality> size{
        {outer_work_iters, inner_work_iters, total_threads}};
    std::array<gko::acc::size_type, dimensionality - 1> stride{
        static_cast<gko::acc::size_type>(grid_.x) * block_.x * inner_work_iters,
        static_cast<gko::acc::size_type>(block_.x)};
    /*/
    static_assert(dimensionality == 1, "Dimensionality must be 1!");
    std::array<gko::acc::size_type, dimensionality> size{{outer_work_iters *
    inner_work_iters * total_threads}};
    //*/
    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    auto acc = range(size, data_ptr, stride);

    timer t;
    t.start();
    benchmark_accessor_kernel<default_block_size, outer_work_iters,
                              inner_work_iters, compute_iters, ArType, range>
        <<<grid_, block_>>>(acc, input);
    t.stop();
    auto kernel_info = get_kernel_info<outer_work_iters, inner_work_iters,
                                       compute_iters, StType>(num_elems);
    return {kernel_info.bytes, kernel_info.comps, t.get_time()};
}

#endif  // BENCHMARK_CUDA_CUH_
