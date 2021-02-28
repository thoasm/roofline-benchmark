#ifndef BENCHMARK_CUDA_CUH_
#define BENCHMARK_CUDA_CUH_

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>

//
#include <array>
#include <cinttypes>
#include <iostream>
#include <type_traits>

#include "../benchmark_info.hpp"
#include "../exec_helper.hpp"

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
void run_benchmark_hand(std::size_t num_elems, T input, T *data) {
    const std::size_t parallel_iters =
        ceildiv(num_elems, inner_work_iters * outer_work_iters);
    const std::int64_t outer_stride = inner_work_iters;
    const std::int64_t parallel_stride = outer_stride * outer_work_iters;
#pragma openmp parallel for schedule(static, 128) simd
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
#if USE_ARRAY
        // TODO: Add USE_ARRAY preprocessor
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<T, inner_work_iters> reg;
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[i] = data[pi * parallel_stride + o * outer_stride + i];
#pragma unroll
                // TODO Problem: can't be unrolled with FMA because of
                //               dependency
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    reg[i] = reg[i] * reg[i] + input;
                }
            }
            T reduced{};
            for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
                reduced = reg[i] * reg[i + 1] + reduced;
            }
            // Is never true, but there to prevent optimization
            if (reduced == static_cast<T>(-1)) {
                data[pi + o * outer_stride] = reduced;
            }
        }
#else
        throw "NOT IMPLEMENTED";
#endif
    }
}

template <typename ArType, typename StType, std::int32_t block_size,
          std::int32_t outer_work_iters, std::int32_t inner_work_iters,
          std::int32_t compute_iters, std::size_t dimensionality>
void run_benchmark_accessor(std::size_t num_elems, ArType input,
                            StType *data_ptr) {
    const std::size_t parallel_iters =
        ceildiv(num_elems, inner_work_iters * outer_work_iters);

    //*
    static_assert(dimensionality == 3, "Dimensionality must be 3!");
    std::array<std::size_t, dimensionality> size{
        {parallel_iters, outer_work_iters, inner_work_iters}};
    /*/
    static_assert(dimensionality == 1, "Dimensionality must be 1!");
    std::array<std::size_t, dimensionality> size{{outer_work_iters *
    inner_work_iters * total_threads}};
    //*/
    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    auto acc = range(size, data_ptr);

#pragma openmp parallel for
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
#if USE_ARRAY
        // TODO: Add USE_ARRAY preprocessor
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<ArType, inner_work_iters> reg;
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[i] = acc(pi, o, i);
#pragma unroll
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    reg[i] = reg[i] * reg[i] + input;
                }
            }
            ArType reduced{};
            for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
                reduced = reg[i] * reg[i + 1] + reduced;
            }
            // Is never true, but there to prevent optimization
            if (reduced == static_cast<ArType>(-1)) {
                acc(pi, o, 0) = reduced;
            }
        }
#else
        throw "NOT IMPLEMENTED";
#endif
    }
}

#endif  // BENCHMARK_CUDA_CUH_
