#pragma once

#include <cinttypes>
#include <random>
#include <string>
#include <typeinfo>

#include "helper.cuh"

#define USE_ARRAY 0

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
__global__ void benchmark_kernel(T input, T *__restrict__ data) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(outer_work_iters > 0, "outer_work_iters must be positive!");
    static_assert(compute_iters >= 0, "compute_iters must be positive or zero!");
    const std::int32_t idx =
        blockIdx.x * block_size * inner_work_iters + threadIdx.x;
    const std::int32_t big_stride = gridDim.x * block_size * inner_work_iters;

#if USE_ARRAY
    static_assert(inner_work_iters % 2 == 0,
                  "inner_work_iters must be dividable by 2!");
    T reg[inner_work_iters];
    for (std::int32_t f = 0; f < outer_work_iters; ++f) {
#pragma unroll
        for (std::int32_t g = 0; g < inner_work_iters; ++g) {
            reg[g] = data[idx + g * block_size + f * big_stride];
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg[g] = reg[g] * reg[g] + input;
            }
        }
        T reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<T>(-1)) {
            data[idx + f * big_stride] = reduced;
        }
    }

#else
    T reg{1};
    for (std::int32_t f = 0; f < outer_work_iters; ++f) {
#pragma unroll
        for (std::int32_t g = 0; g < inner_work_iters; ++g) {
            T mem = data[idx + g * block_size + f * big_stride];
            reg = mem * input + reg;
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg = reg * reg + input;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<T>(-1)) {
            data[idx + f * big_stride] = reg;
        }
    }
#endif  // USE_ARRAY
}

template <typename T>
struct type_to_string {
    static const char *get() { return typeid(T).name(); }
};

#define SPECIALIZE_TYPE_TO_STRING(type_)            \
    template <>                                     \
    struct type_to_string<type_> {                  \
        static const char *get() { return #type_; } \
    }

SPECIALIZE_TYPE_TO_STRING(float);
SPECIALIZE_TYPE_TO_STRING(double);
SPECIALIZE_TYPE_TO_STRING(std::int32_t);
SPECIALIZE_TYPE_TO_STRING(std::int16_t);
#undef SPECIALIZE_TYPE_TO_STRING

struct benchmark_info {
    static constexpr bool use_array{static_cast<bool>(USE_ARRAY)};
    // Template params
    std::string precision;
    std::int32_t block_size;
    std::int32_t outer_work_iters;
    std::int32_t inner_work_iters;
    std::int32_t compute_iters;

    // Details from setup
    std::size_t num_elems;
    dim3 grid;
    dim3 block;
    std::size_t total_threads;

    // Details from computation
    std::size_t computations;
    std::size_t size_bytes;
    double time_ms;

    // helper functions
    double get_giops() const {
        return static_cast<double>(computations) / (time_ms * 1e6);
    }
    double get_bw_gbs() const {
        return static_cast<double>(size_bytes) / (time_ms * 1e6);
    }
};

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
benchmark_info run_benchmark(std::size_t num_elems, T input, T *data_ptr) {
    constexpr int average_iters{5};
    benchmark_info info;

    dim3 block_(block_size);
    dim3 grid_(
        ceildiv(num_elems, inner_work_iters * outer_work_iters * block_size));
    info.total_threads = static_cast<std::size_t>(block_.x) * block_.y *
                         block_.z * grid_.x * grid_.y * grid_.z;
    if (grid_.y != 1 || grid_.z != 1) {
        std::cerr << "Grid is expected to only have x-dimension!\n";
    }
    if (block_.y != 1 || block_.z != 1) {
        std::cerr << "Block is expected to only have x-dimension!\n";
    }

    info.precision = type_to_string<T>::get();
    info.block_size = block_size;
    info.outer_work_iters = outer_work_iters;
    info.inner_work_iters = inner_work_iters;
    info.compute_iters = compute_iters;
    info.num_elems = num_elems;
    info.grid = grid_;
    info.block = block_;
    info.size_bytes = num_elems * sizeof(T);
#if USE_ARRAY
    info.computations = info.total_threads * info.outer_work_iters *
                        info.inner_work_iters *
                        (static_cast<std::size_t>(info.compute_iters) * 2 + 2/2);
    // Note: 2/2(==1) because: 2 for FMA for inner/2 iterations
#else
    info.computations = info.total_threads * info.outer_work_iters *
                        info.inner_work_iters *
                        static_cast<std::size_t>(info.compute_iters + 1) * 2;
#endif
    // Warmup
    benchmark_kernel<T, block_size, outer_work_iters, inner_work_iters,
                     compute_iters><<<grid_, block_>>>(input, data_ptr);
    CUDA_CALL(cudaDeviceSynchronize());

    double time_{0};
    for (int i = 0; i < average_iters; ++i) {
        cuda_timer timer_;
        timer_.start();
        benchmark_kernel<T, block_size, outer_work_iters, inner_work_iters,
                         compute_iters><<<grid_, block_>>>(input, data_ptr);
        timer_.stop();
        time_ += timer_.get_time();
    }
    info.time_ms = time_ / static_cast<double>(average_iters);
    return info;
}
