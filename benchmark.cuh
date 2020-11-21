#pragma once

#include <cinttypes>
#include <random>

#include "helper.cuh"

#define USE_ARRAY 1

template <typename T, std::int32_t block_size, std::int32_t granularity,
          std::int32_t fusion_degree, std::int32_t compute_iter>
__global__ void benchmark_kernel(T summand, T *__restrict__ data) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(fusion_degree > 0, "fusion_degree must be positive!");
    static_assert(compute_iter > 0, "compute_iter must be positive!");
    static_assert(granularity % 2 == 0, "granularity must be dividable by 2!");
    const std::int32_t idx =
        blockIdx.x * block_size * granularity + threadIdx.x;
    const std::int32_t big_stride = gridDim.x * block_size * granularity;

#if USE_ARRAY
    T reg[granularity];
    for (std::int32_t f = 0; f < fusion_degree; ++f) {
#pragma unroll
        for (std::int32_t g = 0; g < granularity; ++g) {
            reg[g] = data[idx + g * block_size + f * big_stride];
#pragma unroll
            for (std::int32_t c = 0; c < compute_iter; ++c) {
                reg[g] = reg[g] * reg[g] + summand;
            }
        }
        T reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < granularity; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<T>(-1)) {
            data[idx + f * big_stride] = reduced;
        }
    }

#else
    T reg{};
    for (std::int32_t f = 0; f < fusion_degree; ++f) {
#pragma unroll
        for (std::int32_t g = 0; g < granularity; ++g) {
            T mem = data[idx + g * block_size + f * big_stride];
            reg = reg * mem + summand;
#pragma unroll
            for (std::int32_t c = 0; c < compute_iter; ++c) {
                reg = reg * reg + summand;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<T>(-1)) {
            data[idx + f * big_stride] = reg;
        }
    }
#endif
}

template <typename T, std::int32_t block_size, std::int32_t granularity,
          std::int32_t fusion_degree, std::int32_t compute_iter>
struct benchmark {
   public:
    benchmark() = delete;
    benchmark(std::size_t num_elems)
        : num_elems_(num_elems),
          block_(block_size),
          grid_(ceildiv(num_elems_, granularity * fusion_degree * block_size)),
          latest_time_ms_{0} {
        if (grid_.y != 1 || grid_.z != 1) {
            std::cerr << "Grid is expected to only have x-dimension!\n";
        }
        if (block_.y != 1 || block_.z != 1) {
            std::cerr << "Block is expected to only have x-dimension!\n";
        }
        total_threads_ = static_cast<std::size_t>(block_.x) * block_.y *
                         block_.z * grid_.x * grid_.y * grid_.z;
    }

    // returns GFLOPs
    double get_flops() const {
#if USE_ARRAY
        std::size_t comps = total_threads_ * fusion_degree * granularity *
                            (static_cast<std::size_t>(compute_iter) * 2 + 1);
#else
        std::size_t comps = total_threads_ * fusion_degree * granularity *
                            static_cast<std::size_t>(compute_iter + 1) * 2;
#endif
        return static_cast<double>(comps) / (latest_time_ms_ * 1e6);
    }

    // returns bandwidth in GB/s
    double get_bandwidth() const {
        auto bytes = num_elems_ * sizeof(T);
        // std::cout << "MemorySize: " << bytes << '\n';
        return static_cast<double>(bytes) / (latest_time_ms_ * 1e6);
    }

    double run(T summand, T *data_ptr) {
        timer_.reset();
        CUDA_CALL(cudaDeviceSynchronize());

        timer_.start();
        benchmark_kernel<T, block_size, granularity, fusion_degree,
                         compute_iter><<<grid_, block_>>>(summand, data_ptr);
        timer_.stop();
        latest_time_ms_ = timer_.get_time();
        return latest_time_ms_;
    }

   private:
    std::size_t num_elems_;
    dim3 block_;
    dim3 grid_;
    std::size_t total_threads_;
    cuda_timer timer_;
    double latest_time_ms_;
};
