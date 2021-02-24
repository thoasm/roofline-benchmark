#ifndef BENCHMARK_INFO_HPP_
#define BENCHMARK_INFO_HPP_

#include <cinttypes>
#include <cstddef>  // for std::size_t

#define USE_ARRAY 1

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

    void calculate_computations() {
#if USE_ARRAY
        computations = total_threads * outer_work_iters * inner_work_iters *
                       (static_cast<std::size_t>(compute_iters) * 2 + 2 / 2);
        // Note: 2/2(==1) because: 2 for FMA for inner/2 iterations
#else
        computations = total_threads * outer_work_iters * inner_work_iters *
                       static_cast<std::size_t>(compute_iters + 1) * 2;
#endif
    }
};

#endif  // BENCHMARK_INFO_HPP_
