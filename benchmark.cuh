#pragma once

#include <cinttypes>
#include <core/base/accessors.hpp>
#include <ginkgo/core/base/range.hpp>
#include <random>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "helper.cuh"

#define USE_ARRAY 0
#define USE_ACCESSOR 0

/**
 * Note: when changing the `input` type to int, followed by casting it to `T` /
 * `value_type`, both the Accessor and the Pointer implementation are
 * significantly faster for double computations 128 and 256 compared to keeping
 * `T input`.
 * Currently, it is unclear why that happens!
 */

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
__global__ void benchmark_kernel(const T input, T *__restrict__ data) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(outer_work_iters > 0, "outer_work_iters must be positive!");
    static_assert(compute_iters >= 0,
                  "compute_iters must be positive or zero!");
    /*
    const std::int32_t idx =
        blockIdx.x * block_size * inner_work_iters + threadIdx.x;
    const std::int32_t inner_stride = block_size;
    const std::int32_t outer_stride = gridDim.x * block_size * inner_work_iters;
    /*/
    const std::int32_t idx = blockIdx.x * block_size + threadIdx.x;
    const std::int32_t inner_stride = gridDim.x * block_size;
    const std::int32_t outer_stride = inner_work_iters * inner_stride;
    //*/
    // const T input = static_cast<T>(i_input);

#if USE_ARRAY
    static_assert(inner_work_iters % 2 == 0,
                  "inner_work_iters must be dividable by 2!");
    T reg[inner_work_iters];
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            reg[i] = data[idx + i * inner_stride + o * outer_stride];
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg[i] = reg[i] * reg[i] + input;
            }
        }
        T reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<T>(-1)) {
            data[idx + o * outer_stride] = reduced;
        }
    }

#else
    T reg{1};
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            const T mem = data[idx + i * inner_stride + o * outer_stride];
            reg = mem * input + reg;
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg = reg * reg + input;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<T>(-1)) {
            data[idx + o * outer_stride] = reg;
        }
    }
#endif  // USE_ARRAY
}

// Specialization for Accessor
template <typename Input, typename Accessor, std::int32_t block_size,
          std::int32_t outer_work_iters, std::int32_t inner_work_iters,
          std::int32_t compute_iters>
__global__ void benchmark_kernel(const Input input, Accessor acc) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(outer_work_iters > 0, "outer_work_iters must be positive!");
    static_assert(compute_iters >= 0,
                  "compute_iters must be positive or zero!");
    /*
    const std::int32_t idx =
        blockIdx.x * block_size * inner_work_iters + threadIdx.x;
    const std::int32_t outer_stride = gridDim.x * block_size * inner_work_iters;
    /*/
    const std::int32_t idx = blockIdx.x * block_size + threadIdx.x;

    using value_type = typename Accessor::accessor::arithmetic_type;
    static_assert(std::is_same<value_type, Input>::value, "Types must match!");

    // const auto input = static_cast<value_type>(i_input);

#if USE_ARRAY
    static_assert(inner_work_iters % 2 == 0,
                  "inner_work_iters must be dividable by 2!");
    value_type reg[inner_work_iters];
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            reg[i] = acc(o, i, idx);
            // reg[i] = acc(idx + i * block_size + o * outer_stride);
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg[i] = reg[i] * reg[i] + input;
            }
        }
        value_type reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<value_type>(-1)) {
            // acc(idx + o * outer_stride) = reduced;
            acc(o, 0, idx) = reduced;
        }
    }

#else
    value_type reg{1};
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            // value_type mem = acc(idx + i * block_size + o * outer_stride);
            const value_type mem = acc(o, i, idx);
            reg = mem * input + reg;
#pragma unroll
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg = reg * reg + input;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<value_type>(-1)) {
            // acc(idx + o * outer_stride) = reg;
            acc(o, 0, idx) = reg;
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
    info.computations =
        info.total_threads * info.outer_work_iters * info.inner_work_iters *
        (static_cast<std::size_t>(info.compute_iters) * 2 + 2 / 2);
    // Note: 2/2(==1) because: 2 for FMA for inner/2 iterations
#else
    info.computations = info.total_threads * info.outer_work_iters *
                        info.inner_work_iters *
                        static_cast<std::size_t>(info.compute_iters + 1) * 2;
#endif

    auto i_input = static_cast<std::int32_t>(input);
    gko::dim<3> size{outer_work_iters, inner_work_iters, info.total_threads};
    // std::array<std::size_t, 2> stride{static_cast<std::size_t>(grid_.x *
    // block_size * inner_work_iters), block_size};
    using accessor = gko::accessor::reduced_row_major<3, T, T>;
    using range = gko::range<accessor>;
    auto acc = range(size, data_ptr /*, stride*/);
    // Warmup
#if USE_ACCESSOR
    benchmark_kernel<T, range, block_size, outer_work_iters, inner_work_iters,
                     compute_iters><<<grid_, block_>>>(input, acc);
#else
    benchmark_kernel<T, block_size, outer_work_iters, inner_work_iters,
                     compute_iters><<<grid_, block_>>>(input, data_ptr);
#endif
    CUDA_CALL(cudaDeviceSynchronize());

    double time_{0};
    for (int i = 0; i < average_iters; ++i) {
        cuda_timer timer_;
        timer_.start();
#if USE_ACCESSOR
        benchmark_kernel<T, range, block_size, outer_work_iters,
                         inner_work_iters, compute_iters>
            <<<grid_, block_>>>(input, acc);
#else
        benchmark_kernel<T, block_size, outer_work_iters, inner_work_iters,
                         compute_iters><<<grid_, block_>>>(input, data_ptr);
#endif
        timer_.stop();
        time_ += timer_.get_time();
    }
    info.time_ms = time_ / static_cast<double>(average_iters);
    return info;
}
