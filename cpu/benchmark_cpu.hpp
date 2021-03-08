#ifndef BENCHMARK_CUDA_CUH_
#define BENCHMARK_CUDA_CUH_

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>

//
#include <immintrin.h>

#include <array>
#include <cinttypes>
#include <iostream>
#include <type_traits>

#include "../benchmark_info.hpp"
#include "../exec_helper.hpp"

//

#define SINGLE_COMPUTATION true

// fakes reading from p (when p is an address, it forces the object to have an
// address) and writing / touching all available memory
static void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }

// fakes reading and writing all available memory
static void clobber() { asm volatile("" : : : "memory"); }

//

template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
std::size_t run_benchmark_hand(const std::size_t num_elems, const T input,
                               T *data) {
    const std::size_t parallel_iters =
        ceildiv(num_elems, inner_work_iters * outer_work_iters);
    const std::int64_t outer_stride = inner_work_iters;
    const std::int64_t parallel_stride = outer_stride * outer_work_iters;
#if SINGLE_COMPUTATION
#pragma omp parallel for
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            static_assert(inner_work_iters % 2 == 0,
                          "inner_work_iters must be a multiple of 2!");
            std::array<T, inner_work_iters> reg;
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[i] = data[pi * parallel_stride + o * outer_stride + i];
            }
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                //#pragma unroll
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
    }
    return num_elems * (static_cast<std::size_t>(compute_iters) * 2 + 2 / 2);
#else
#pragma omp parallel for
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        /*
        // Hand-vectorized version. ONLY works with double !!!
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            static_assert(inner_work_iters % 2 == 0,
                          "inner_work_iters must be a multiple of 2!");
            using vec_double = __m256d;
            // const vec_double v_input = _mm256_broadcast_sd(&input);
            const vec_double v_input = _mm256_set1_pd(input);
            vec_double reg0{}, reg1{};
            double reduced{};
            for (std::int32_t i = 0; i < inner_work_iters; i += 8) {
                reg0 = _mm256_loadu_pd(data + pi * parallel_stride +
                                       o * outer_stride + i);
                reg1 = _mm256_loadu_pd(data + pi * parallel_stride +
                                       o * outer_stride + i + 4);
                // reg = mem * input + reg;
                //#pragma unroll
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    reg0 = _mm256_fmadd_pd(reg0, reg0, v_input);
                    reg1 = _mm256_fmadd_pd(reg1, reg1, v_input);
                    // reg = reg * reg + input;
                }
                union tmp {
                    double d[4];
                    vec_double vd;
                } conv;
                conv.vd = reg0;
                reduced = conv.d[0] * conv.d[1] + reduced;
                reduced = conv.d[2] * conv.d[3] + reduced;
                conv.vd = reg1;
                reduced = conv.d[0] * conv.d[1] + reduced;
                reduced = conv.d[2] * conv.d[3] + reduced;
                // reduced = _mm256_fmadd_pd(reg0, reg1, reduced);
            }
            // Is never true, but there to prevent optimization
            if (reduced == -1) {
                data[pi + o * outer_stride] = reduced;
            }
        }
        */
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            static_assert(inner_work_iters % 2 == 0,
                          "inner_work_iters must be a multiple of 2!");
            std::array<T, inner_work_iters> reg0;
            std::array<T, inner_work_iters> reg1;
            std::array<T, inner_work_iters> reg2;
            for (std::int32_t i = 0; i < inner_work_iters; i += 1) {
                reg0[i] = data[pi * parallel_stride + o * outer_stride + i];
                reg1[i] = reg0[i];
                reg2[i] = reg0[i];
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    reg0[i] = reg0[i] * reg0[i] + input;
                    reg1[i] = input * reg1[i] + reg1[i];
                    reg2[i] = reg2[i] * reg2[i] + reg2[i];
                }
            }
            T reduced{};
            for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
                reduced = reg0[i] * reg0[i + 1] + reduced;
                reduced = reg1[i] * reg1[i + 1] + reduced;
                reduced = reg2[i] * reg2[i + 1] + reduced;
            }
            // Is never true, but there to prevent optimization
            if (reduced == static_cast<T>(-1)) {
                data[pi * parallel_stride + o * outer_stride] = reduced;
            }
        }
    }
    return num_elems * 3 *
           (static_cast<std::size_t>(compute_iters) * 2 + 2 / 2);
#endif
}

template <typename ArType, typename StType, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters,
          std::size_t dimensionality>
std::size_t run_benchmark_accessor(const std::size_t num_elems,
                                   const ArType input, StType *data_ptr) {
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

#if SINGLE_COMPUTATION
#pragma omp parallel for
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            static_assert(inner_work_iters % 2 == 0,
                          "inner_work_iters must be a multiple of 2!");
            std::array<ArType, inner_work_iters> reg;
            // By writing all data first, the additional optimization
            // that performs 2 FMA simultaneously for <double, float> is
            // prevented
            for (std::int32_t i = 0; i < inner_work_iters; i += 1) {
                reg[i] = acc(pi, o, i);
            }
            for (std::int32_t i = 0; i < inner_work_iters; i += 1) {
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
    }
    return num_elems * (static_cast<std::size_t>(compute_iters) * 2 + 2 / 2);
#else
#pragma omp parallel for
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            static_assert(inner_work_iters % 2 == 0,
                          "inner_work_iters must be a multiple of 2!");
            std::array<ArType, inner_work_iters> reg0;
            std::array<ArType, inner_work_iters> reg1;
            std::array<ArType, inner_work_iters> reg2;
            for (std::int32_t i = 0; i < inner_work_iters; i += 1) {
                reg0[i] = acc(pi, o, i);
                reg1[i] = reg0[i];
                reg2[i] = reg0[i];
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    reg0[i] = reg0[i] * reg0[i] + input;
                    reg1[i] = input * reg1[i] + reg1[i];
                    reg2[i] = reg2[i] * reg2[i] + reg2[i];
                }
            }
            ArType reduced{};
            for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
                reduced = reg0[i] * reg0[i + 1] + reduced;
                reduced = reg1[i] * reg1[i + 1] + reduced;
                reduced = reg2[i] * reg2[i + 1] + reduced;
            }
            // Is never true, but there to prevent optimization
            if (reduced == static_cast<ArType>(-1)) {
                acc(pi, o, 0) = reduced;
            }
        }
    }
    return num_elems * 3 *
           (static_cast<std::size_t>(compute_iters) * 2 + 2 / 2);
#endif
}

#endif  // BENCHMARK_CUDA_CUH_
