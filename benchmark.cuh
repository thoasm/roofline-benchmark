#ifndef BENCHMARK_CUH_
#define BENCHMARK_CUH_

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <array>
#include <cinttypes>
#include <random>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "benchmark_cuda.cuh"
#include "benchmark_info.hpp"
#include "helper.cuh"

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

enum class Precision { Pointer, AccessorKeep, AccessorReduced };

template <typename T, std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
benchmark_info run_benchmark(std::size_t num_elems, T input, T *data_ptr,
                             Precision prec = Precision::Pointer) {
    constexpr int average_iters{5};
    benchmark_info info;
    // Precision prec = Precision::Pointer;

    info.total_threads =
        ceildiv(num_elems, inner_work_iters * outer_work_iters * block_size) *
        block_size;

    info.precision = type_to_string<T>::get();
    info.block_size = block_size;
    info.outer_work_iters = outer_work_iters;
    info.inner_work_iters = inner_work_iters;
    info.compute_iters = compute_iters;
    info.num_elems = num_elems;
    info.size_bytes = num_elems * sizeof(T);

    info.calculate_computations();

    constexpr std::size_t dimensionality{3};

    auto run_hand_kernel = [&]() {
        run_benchmark_hand<T, block_size, outer_work_iters, inner_work_iters,
                           compute_iters>(num_elems, input, data_ptr);
    };
    auto run_accessor_kernel = [&]() {
        run_benchmark_accessor<T, T, block_size, outer_work_iters,
                               inner_work_iters, compute_iters, dimensionality>(
            num_elems, input, data_ptr);
    };
    using lower_precision = std::conditional_t<
        std::is_same<T, double>::value, float,
        std::conditional_t<std::is_same<T, std::int32_t>::value,
                           std::int16_t, T>>;
    auto lower_ptr = reinterpret_cast<lower_precision *>(data_ptr);
    auto run_lower_accessor_kernel = [&]() {
        run_benchmark_accessor<T, lower_precision, block_size, outer_work_iters,
                               inner_work_iters, compute_iters, dimensionality>(
            num_elems, input, lower_ptr);
    };

    // auto i_input = static_cast<std::int32_t>(input);
    double time_{0};
    if (prec == Precision::Pointer) {
        run_hand_kernel();
        CUDA_CALL(cudaDeviceSynchronize());

        for (int i = 0; i < average_iters; ++i) {
            cuda_timer timer_;
            timer_.start();
            run_hand_kernel();
            timer_.stop();
            time_ += timer_.get_time();
        }
    } else if (prec == Precision::AccessorKeep) {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + typeid(T).name() + ", " + typeid(T).name() +
                         ">";
        run_accessor_kernel();
        CUDA_CALL(cudaDeviceSynchronize());

        for (int i = 0; i < average_iters; ++i) {
            cuda_timer timer_;
            timer_.start();
            run_accessor_kernel();
            timer_.stop();
            time_ += timer_.get_time();
        }
    } else {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + typeid(T).name() + ", " +
                         typeid(lower_precision).name() + ">";
        info.size_bytes = info.num_elems * sizeof(lower_precision);
        
        // Warmup
        run_lower_accessor_kernel();
        CUDA_CALL(cudaDeviceSynchronize());

        for (int i = 0; i < average_iters; ++i) {
            cuda_timer timer_;
            timer_.start();
            run_lower_accessor_kernel();
            timer_.stop();
            time_ += timer_.get_time();
        }
    }
    info.time_ms = time_ / static_cast<double>(average_iters);
    return info;
}

#endif  // BENCHMARK_CUH_
