#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <array>
#include <cinttypes>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "benchmark_info.hpp"
#include "config.hpp"
#include "exec_helper.hpp"

#if ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_CUDA
#include "cuda/benchmark_cuda.cuh"
#elif ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_HIP
#include "hip/benchmark_hip.hip.hpp"
#elif ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_CPU
#include "cpu/benchmark_cpu.hpp"
#endif

template <typename T>
struct type_to_string {
    static const char *get() { return typeid(T).name(); }
};

#define SPECIALIZE_TYPE_TO_STRING(type_)            \
    template <>                                     \
    struct type_to_string<type_> {                  \
        static const char *get() { return #type_; } \
    }

#define SPECIALIZE2_TYPE_TO_STRING(type_, str_)   \
    template <>                                   \
    struct type_to_string<type_> {                \
        static const char *get() { return str_; } \
    }

SPECIALIZE_TYPE_TO_STRING(float);
SPECIALIZE_TYPE_TO_STRING(double);
SPECIALIZE2_TYPE_TO_STRING(std::int32_t, "int32");
SPECIALIZE2_TYPE_TO_STRING(std::int16_t, "int16");
#undef SPECIALIZE_TYPE_TO_STRING
#undef SPECIALIZE2_TYPE_TO_STRING

enum class Precision { Pointer, AccessorKeep, AccessorReduced };

template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
benchmark_info run_benchmark(std::size_t num_elems, T input, T *data_ptr,
                             Precision prec = Precision::Pointer) {
    constexpr int average_iters{5};
    benchmark_info info;
    // Precision prec = Precision::Pointer;

    info.precision = type_to_string<T>::get();
    info.outer_work_iters = outer_work_iters;
    info.inner_work_iters = inner_work_iters;
    info.compute_iters = compute_iters;
    info.num_elems = num_elems;
    info.size_bytes = num_elems * sizeof(T);

    constexpr std::size_t dimensionality{3};

    auto run_hand_kernel = [&]() {
        return run_benchmark_hand<T, outer_work_iters, inner_work_iters,
                                  compute_iters>(num_elems, input, data_ptr);
    };
    auto run_accessor_kernel = [&]() {
        return run_benchmark_accessor<T, T, outer_work_iters, inner_work_iters,
                                      compute_iters, dimensionality>(
            num_elems, input, data_ptr);
    };
    using lower_precision = std::conditional_t<
        std::is_same<T, double>::value, float,
        std::conditional_t<std::is_same<T, std::int32_t>::value, std::int16_t,
                           T>>;
    auto lower_ptr = reinterpret_cast<lower_precision *>(data_ptr);
    auto run_lower_accessor_kernel = [&]() {
        return run_benchmark_accessor<T, lower_precision, outer_work_iters,
                                      inner_work_iters, compute_iters,
                                      dimensionality>(num_elems, input,
                                                      lower_ptr);
    };

    // auto i_input = static_cast<std::int32_t>(input);
    double time_{0};
    timer t;
    if (prec == Precision::Pointer) {
        info.computations = run_hand_kernel();
        synchronize();

        for (int i = 0; i < average_iters; ++i) {
            t.start();
            run_hand_kernel();
            t.stop();
            time_ += t.get_time();
            t.reset();
        }
    } else if (prec == Precision::AccessorKeep) {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + typeid(T).name() + ", " + typeid(T).name() +
                         ">";
        info.computations = run_accessor_kernel();
        synchronize();

        for (int i = 0; i < average_iters; ++i) {
            t.start();
            run_accessor_kernel();
            t.stop();
            time_ += t.get_time();
            t.reset();
        }
    } else {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + typeid(T).name() + ", " +
                         typeid(lower_precision).name() + ">";
        info.size_bytes = info.num_elems * sizeof(lower_precision);

        // Warmup
        info.computations = run_lower_accessor_kernel();
        synchronize();

        for (int i = 0; i < average_iters; ++i) {
            t.start();
            run_lower_accessor_kernel();
            t.stop();
            time_ += t.get_time();
            t.reset();
        }
    }
    info.time_ms = time_ / static_cast<double>(average_iters);
    return info;
}

#endif  // BENCHMARK_HPP_
