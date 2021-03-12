#ifndef BENCHMARK_INFO_HPP_
#define BENCHMARK_INFO_HPP_

#include <cinttypes>
#include <cstddef>  // for std::size_t
#include <string>

struct kernel_runtime_info {
    std::size_t bytes;
    std::size_t comps;
    double runtime_ms;
};

struct benchmark_info {
    // Template params
    std::string precision;
    std::int32_t outer_work_iters;
    std::int32_t inner_work_iters;
    std::int32_t compute_iters;

    // Details from setup
    std::size_t num_elems;

    // Details from computation
    std::size_t computations;
    std::size_t memory_moved_bytes;
    double time_ms;

    // helper functions
    double get_giops() const {
        return static_cast<double>(computations) / (time_ms * 1e6);
    }
    double get_bw_gbs() const {
        return static_cast<double>(memory_moved_bytes) / (time_ms * 1e6);
    }
    void set_kernel_info(kernel_runtime_info res) {
        memory_moved_bytes = res.bytes;
        computations = res.comps;
    }
};

#endif  // BENCHMARK_INFO_HPP_
