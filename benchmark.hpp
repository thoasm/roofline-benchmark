#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include <accessor/posit.hpp>
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <array>
#include <cinttypes>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

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
    static const char* get() { return typeid(T).name(); }
    static const char* get_short() { return typeid(T).name(); }
};

#define SPECIALIZE_TYPE_TO_STRING(type_, str_, sh_str_)    \
    template <>                                            \
    struct type_to_string<type_> {                         \
        static const char* get() { return str_; }          \
        static const char* get_short() { return sh_str_; } \
    }

SPECIALIZE_TYPE_TO_STRING(float, "float", "f");
SPECIALIZE_TYPE_TO_STRING(double, "double", "d");
SPECIALIZE_TYPE_TO_STRING(std::int32_t, "int32", "i");
SPECIALIZE_TYPE_TO_STRING(std::int16_t, "int16", "s");
SPECIALIZE_TYPE_TO_STRING(gko::acc::posit32_2, "posit<32,2>", "p32");
SPECIALIZE_TYPE_TO_STRING(gko::acc::posit32_3, "posit<32,3>", "p32-3");
SPECIALIZE_TYPE_TO_STRING(gko::acc::posit16_2, "posit<16,2>", "p16");
SPECIALIZE_TYPE_TO_STRING(char, "frsz2-32", "frsz2-32");
#undef SPECIALIZE_TYPE_TO_STRING

enum class Precision {
    Pointer,
    AccessorKeep,
    AccessorReduced,
    AccessorPosit,
    Frsz2_32
};

class time_series {
public:
    using time_format = double;
    time_series() { series.reserve(30); }

    void add_time(time_format time) { series.push_back(time); }

    time_format get_time() const
    {
        auto reduced{std::numeric_limits<time_format>::max()};
        for (const auto& t : series) {
            if (t < reduced) {
                reduced = t;
            }
        }
        return reduced;
    }

private:
    std::vector<time_format> series;
};

template <typename T, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters>
benchmark_info run_benchmark(std::size_t num_elems, memory& data, unsigned seed,
                             RandomNumberGenerator& rng,
                             Precision prec = Precision::Pointer)
{
    // Reallocate if data is on the CPU to ensure that the correct cores get the
    // data
#if ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_CPU
    data.re_allocate();
#endif

    auto data_ptr = data.template get<T>();
    constexpr int number_runs{20};
    benchmark_info info;
    // Precision prec = Precision::Pointer;

    info.precision = type_to_string<T>::get();
    info.outer_work_iters = outer_work_iters;
    info.inner_work_iters = inner_work_iters;
    info.compute_iters = compute_iters;
    info.num_elems = num_elems;

    using lower_precision = std::conditional_t<
        std::is_same<T, double>::value, float,
        std::conditional_t<std::is_same<T, std::int32_t>::value, std::int16_t,
                           T>>;
    using posit_precision =
        std::conditional_t<std::is_same<T, double>::value, gko::acc::posit32_2,
                           std::conditional_t<std::is_same<T, float>::value,
                                              gko::acc::posit16_2, T>>;
    using frsz_32 = frsz::frsz2_compressor<32, 32, double, std::int16_t>;
    auto lower_ptr = data.template get<lower_precision>();
    auto posit_ptr = data.template get<posit_precision>();
    auto byte_ptr = data.template get<std::uint8_t>();

    const auto input = static_cast<T>(seed);

    constexpr std::size_t dimensionality{3};
    auto run_set_data = [&]() {
        return set_data<T>(num_elems, data_ptr, seed, rng);
    };
    auto run_set_data_lower = [&]() {
        return set_data<lower_precision>(num_elems, lower_ptr, seed, rng);
    };
    auto run_set_data_posit = [&]() {
        return set_data<posit_precision>(num_elems, posit_ptr, seed, rng);
    };
    auto run_set_data_frsz2 = [&]() {
        return set_frsz2_data<frsz_32>(num_elems, byte_ptr, seed, rng);
    };

    auto run_hand_kernel = [&]() {
        return run_benchmark_hand<T, outer_work_iters, inner_work_iters,
                                  compute_iters>(num_elems, data_ptr, input);
    };
    auto run_accessor_kernel = [&]() {
        return run_benchmark_accessor<T, T, outer_work_iters, inner_work_iters,
                                      compute_iters, dimensionality>(
            num_elems, data_ptr, input);
    };
    auto run_lower_accessor_kernel = [&]() {
        return run_benchmark_accessor<T, lower_precision, outer_work_iters,
                                      inner_work_iters, compute_iters,
                                      dimensionality>(num_elems, lower_ptr,
                                                      input);
    };
    auto run_posit_accessor_kernel = [&]() {
        return run_benchmark_accessor<T, posit_precision, outer_work_iters,
                                      inner_work_iters, compute_iters,
                                      dimensionality>(num_elems, posit_ptr,
                                                      input);
    };
    auto run_frsz_kernel = [&]() {
        return run_benchmark_frsz<frsz_32, outer_work_iters, inner_work_iters,
                                  compute_iters>(num_elems, byte_ptr, input);
    };

    // auto i_input = static_cast<std::int32_t>(input);
    time_series t_series;
    if (prec == Precision::Pointer) {
        run_set_data();
        synchronize();

        /*
        if (std::is_same<T, double>::value && outer_work_iters == 4 &&
            inner_work_iters == 8 &&
            (compute_iters == 0 || compute_iters == 1)) {
            auto vals = data.template get_vector<double>();
            for (std::size_t i = data.get_num_elems() - 20;
                 i < data.get_num_elems(); ++i) {
                std::cout << vals[i] << '\n';
            }
        }
        //*/
        // Warmup
        info.set_kernel_info(run_hand_kernel());
        synchronize();

        for (int i = 0; i < number_runs; ++i) {
            auto res = run_hand_kernel();
            t_series.add_time(res.runtime_ms);
            synchronize();
        }
    } else if (prec == Precision::AccessorKeep) {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + type_to_string<T>::get_short() + ", " +
                         type_to_string<T>::get_short() + ">";
        run_set_data();
        synchronize();
        // Warmup
        info.set_kernel_info(run_accessor_kernel());
        synchronize();

        for (int i = 0; i < number_runs; ++i) {
            auto res = run_accessor_kernel();
            t_series.add_time(res.runtime_ms);
            synchronize();
        }
    } else if (prec == Precision::AccessorReduced) {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + type_to_string<T>::get_short() + ", " +
                         type_to_string<lower_precision>::get_short() + ">";

        run_set_data_lower();
        synchronize();
        // Warmup
        info.set_kernel_info(run_lower_accessor_kernel());
        synchronize();

        for (int i = 0; i < number_runs; ++i) {
            auto res = run_lower_accessor_kernel();
            t_series.add_time(res.runtime_ms);
            synchronize();
        }
    } else if (prec == Precision::AccessorPosit) {
        info.precision = std::string("Ac<") + std::to_string(dimensionality) +
                         ", " + type_to_string<T>::get_short() + ", " +
                         type_to_string<posit_precision>::get_short() + ">";

        run_set_data_posit();
        synchronize();
        // Warmup
        info.set_kernel_info(run_posit_accessor_kernel());
        synchronize();

        for (int i = 0; i < number_runs; ++i) {
            auto res = run_posit_accessor_kernel();
            t_series.add_time(res.runtime_ms);
            synchronize();
        }
    } else if (prec == Precision::Frsz2_32) {
        info.precision = std::string("frsz2-32");

        run_set_data_frsz2();
        synchronize();
        // Warmup
        info.set_kernel_info(run_frsz_kernel());
        synchronize();

        for (int i = 0; i < number_runs; ++i) {
            auto res = run_frsz_kernel();
            t_series.add_time(res.runtime_ms);
            synchronize();
        }
    } else {
        info.precision = "error";
    }
    info.time_ms = t_series.get_time();
    return info;
}

#endif  // BENCHMARK_HPP_
