#ifndef BENCHMARK_INFO_HPP_
#define BENCHMARK_INFO_HPP_

#include <cinttypes>
#include <cstddef>  // for std::size_t
#include <string>

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

#endif  // BENCHMARK_INFO_HPP_
