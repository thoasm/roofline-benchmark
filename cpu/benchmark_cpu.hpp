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

//#define READ_WRITE_BENCHMARK true
#define READ_WRITE_BENCHMARK false

#define PARALLEL_FOR_SCHEDULE schedule(static, 1024)
//#define PARALLEL_FOR_SCHEDULE
constexpr std::int32_t num_parallel_computations{4};

//

template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
void set_data(const std::size_t num_elems, const T input, T *data) {
    const std::size_t parallel_iters =
        ceildiv(num_elems, inner_work_iters * outer_work_iters);
    const std::int64_t outer_stride = inner_work_iters;
    const std::int64_t parallel_stride = outer_stride * outer_work_iters;
#pragma omp parallel for PARALLEL_FOR_SCHEDULE
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                data[pi * parallel_stride + o * outer_stride + i] = input;
            }
        }
    }
}

template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
kernel_runtime_info run_benchmark_hand(const std::size_t num_elems,
                                       const T input, T *data) {
    const std::size_t parallel_iters =
        ceildiv(num_elems, inner_work_iters * outer_work_iters);
    const std::int64_t outer_stride = inner_work_iters;
    const std::int64_t parallel_stride = outer_stride * outer_work_iters;
    timer t;
#if READ_WRITE_BENCHMARK
    t.start();
#pragma omp parallel for PARALLEL_FOR_SCHEDULE
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<T, num_parallel_computations> reg{};
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[0] = data[pi * parallel_stride + o * outer_stride + i];
                for (std::int32_t nc = 1; nc < num_parallel_computations;
                     ++nc) {
                    reg[nc] = reg[0] + nc * input;
                }
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    for (std::int32_t nc = 0; nc < num_parallel_computations;
                         ++nc) {
                        reg[nc] = reg[nc] * reg[nc] + input;
                    }
                }
                for (std::int32_t nc = num_parallel_computations - 1; nc > 0;
                     --nc) {
                    reg[nc - 1] = reg[nc - 1] * reg[nc] + input;
                }
                data[pi * parallel_stride + o * outer_stride + i] = reg[0];
            }
            /*
            // Original benchmark code for CPU without num_parallel_computations
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
            // Write output to ensure that vectorization is easy
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                data[pi * parallel_stride + o * outer_stride + i] = reg[i];
            }
            */
        }
    }
    t.stop();
    return {2 * num_elems * sizeof(T),
            num_elems * (num_parallel_computations *
                             static_cast<std::size_t>(compute_iters) * 2  // FMA
                         + num_parallel_computations - 1        // + nc*input
                         + (num_parallel_computations - 1) * 2  // FMA reduce
                         ),
            t.get_time()};
#else
    t.start();
#pragma omp parallel for PARALLEL_FOR_SCHEDULE
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<T, num_parallel_computations> reg{};
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[0] = data[pi * parallel_stride + o * outer_stride + i];
                for (std::int32_t nc = 1; nc < num_parallel_computations;
                     ++nc) {
                    reg[nc] = reg[0] + nc * input;
                }
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    for (std::int32_t nc = 0; nc < num_parallel_computations;
                         ++nc) {
                        reg[nc] = reg[nc] * reg[nc] + input;
                    }
                }
                for (std::int32_t nc = num_parallel_computations - 1; nc > 0;
                     --nc) {
                    reg[nc - 1] = reg[nc - 1] * reg[nc] + input;
                }
                // Is never true, but prevents optimization
                if (reg[0] == static_cast<T>(-1)) {
                    data[pi * parallel_stride + o * outer_stride + i] = reg[0];
                }
            }
        }
    }
    t.stop();
    return {1 * num_elems * sizeof(T),
            num_elems * (num_parallel_computations *
                             static_cast<std::size_t>(compute_iters) * 2  // FMA
                         + num_parallel_computations - 1        // + nc*input
                         + (num_parallel_computations - 1) * 2  // FMA reduce
                         ),
            t.get_time()};
#endif
}

template <typename ArType, typename StType, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters,
          std::size_t dimensionality>
kernel_runtime_info run_benchmark_accessor(const std::size_t num_elems,
                                           const ArType input,
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
    timer t;

#if READ_WRITE_BENCHMARK
    t.start();
#pragma omp parallel for PARALLEL_FOR_SCHEDULE
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<ArType, num_parallel_computations> reg{};
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[0] = acc(pi, o, i);
                for (std::int32_t nc = 1; nc < num_parallel_computations;
                     ++nc) {
                    reg[nc] = reg[0] + nc * input;
                }
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    for (std::int32_t nc = 0; nc < num_parallel_computations;
                         ++nc) {
                        reg[nc] = reg[nc] * reg[nc] + input;
                    }
                }
                for (std::int32_t nc = num_parallel_computations - 1; nc > 0;
                     --nc) {
                    reg[nc - 1] = reg[nc - 1] * reg[nc] + input;
                }
                acc(pi, o, i) = reg[0];
            }
        }
    }
    t.stop();
    return {2 * num_elems * sizeof(StType),
            num_elems * (num_parallel_computations *
                             static_cast<std::size_t>(compute_iters) * 2  // FMA
                         + num_parallel_computations - 1        // + nc*input
                         + (num_parallel_computations - 1) * 2  // FMA reduce
                         ),
            t.get_time()};
#else
    t.start();
#pragma omp parallel for PARALLEL_FOR_SCHEDULE
    for (std::size_t pi = 0; pi < parallel_iters; ++pi) {
        for (std::int32_t o = 0; o < outer_work_iters; ++o) {
            std::array<ArType, num_parallel_computations> reg{};
            for (std::int32_t i = 0; i < inner_work_iters; ++i) {
                reg[0] = acc(pi, o, i);
                for (std::int32_t nc = 1; nc < num_parallel_computations;
                     ++nc) {
                    reg[nc] = reg[0] + nc * input;
                }
                for (std::int32_t c = 0; c < compute_iters; ++c) {
                    for (std::int32_t nc = 0; nc < num_parallel_computations;
                         ++nc) {
                        reg[nc] = reg[nc] * reg[nc] + input;
                    }
                }
                for (std::int32_t nc = num_parallel_computations - 1; nc > 0;
                     --nc) {
                    reg[nc - 1] = reg[nc - 1] * reg[nc] + input;
                }
                // is never true, but prevents optimization
                if (reg[0] == static_cast<ArType>(-1)) {
                    acc(pi, o, i) = reg[0];
                }
            }
        }
    }
    t.stop();
    return {1 * num_elems * sizeof(StType),
            num_elems * (num_parallel_computations *
                             static_cast<std::size_t>(compute_iters) * 2  // FMA
                         + num_parallel_computations - 1        // + nc*input
                         + (num_parallel_computations - 1) * 2  // FMA reduce
                         ),
            t.get_time()};
#endif
}

#endif  // BENCHMARK_CUDA_CUH_
