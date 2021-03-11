#include <cinttypes>
#include <iomanip>
#include <iostream>
#include <random>

#include "benchmark.hpp"
#include "benchmark_info.hpp"
#include "exec_helper.hpp"

constexpr char SEP = ';';
constexpr char NL = '\n';

void print_header() {
    // clang-format off
    std::cout << std::right;
    std::cout << std::setw(12) << "Precision"
              << SEP << std::setw(11) << "[GOP/s]"
              << SEP << std::setw(11) << "BW [GB/s]"
              << SEP << std::setw(11) << "time [ms]"
              << SEP << std::setw(13) << "computations"
              << SEP << std::setw(13) << "data [Bytes]"
              << SEP << std::setw(10) << "Outer Its"
              << SEP << std::setw(10) << "Inner Its"
              << SEP << std::setw(9) << "Comp Its"
              << SEP << std::setw(11) << "# Elements"
              << NL;
    // clang-format on
}

void print_info(const benchmark_info &info) {
    // clang-format off
    // std::cout << std::defaultfloat << std::setprecision(5);
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::right;
    std::cout << std::setw(12) << info.precision
              << SEP << std::setw(11) << info.get_giops()
              << SEP << std::setw(11) << info.get_bw_gbs()
              << SEP << std::setw(11) << info.time_ms
              << SEP << std::setw(13) << info.computations
              << SEP << std::setw(13) << info.memory_moved_bytes
              << SEP << std::setw(10) << info.outer_work_iters
              << SEP << std::setw(10) << info.inner_work_iters
              << SEP << std::setw(9) << info.compute_iters
              << SEP << std::setw(11) << info.num_elems
              << NL;
    // clang-format on
}

// Container for types
template <typename... Types>
struct type_list {};

// Container for values
template <typename T, T... Vals>
struct val_list {};

template <typename T, Precision prec>
struct v_type {
    using type = T;
    static constexpr Precision p = prec;
};

// For compute resolution
template <typename T, typename IT, IT outer, IT inner, typename... Args>
void run_benchmark_variations_compute(val_list<IT>, Args... args) {}

template <typename T, typename IT, IT outer, IT inner, IT compute_k,
          IT... rem_compute>
void run_benchmark_variations_compute(val_list<IT, compute_k, rem_compute...>,
                                      std::size_t num_elems,
                                      std::int32_t r_val) {
    using value_type = typename T::type;
    auto prec = T::p;
    {
        memory<value_type> data(num_elems);
        // Note: Memory is not initialized in order to get put to the correct
        //       core. However, it is possible that optimizations might require
        //       proper initialization.
        auto info = run_benchmark<value_type, outer, inner, compute_k>(
            num_elems, static_cast<value_type>(r_val), data.get(), prec);
        print_info(info);
    }
    // recursion
    run_benchmark_variations_compute<T, IT, outer, inner>(
        val_list<IT, rem_compute...>{}, num_elems, r_val);
}

// For inner resolution
template <typename T, typename IT, IT outer, typename... Args>
void run_benchmark_variations_inner(val_list<IT>, Args... args) {}

template <typename T, typename IT, IT outer, IT inner_k, IT... rem_inner,
          typename... Args>
void run_benchmark_variations_inner(val_list<IT, inner_k, rem_inner...>,
                                    Args... args) {
    run_benchmark_variations_compute<T, IT, outer, inner_k>(args...);
    // recursion
    run_benchmark_variations_inner<T, IT, outer>(val_list<IT, rem_inner...>{},
                                                 args...);
}

// For outer resolution
template <typename T, typename IT, typename... Args>
void run_benchmark_variations_outer(val_list<IT>, Args...) {}

template <typename T, typename IT, IT outer_k, IT... rem_outer,
          typename... Args>
void run_benchmark_variations_outer(val_list<IT, outer_k, rem_outer...>,
                                    Args... args) {
    run_benchmark_variations_inner<T, IT, outer_k>(args...);
    // recursion
    run_benchmark_variations_outer<T>(val_list<IT, rem_outer...>{}, args...);
}

template <typename... Args>
void run_benchmark_variations(type_list<>, Args...) {}

template <typename T, typename... RemTypes, typename... Args>
void run_benchmark_variations(type_list<T, RemTypes...>, Args... args) {
    run_benchmark_variations_outer<T>(args...);
    // recursion
    run_benchmark_variations(type_list<RemTypes...>{}, args...);
}

int main() {
    using i_type = std::int32_t;
    constexpr std::size_t num_elems = 256 * 1024 * 1024;

    /*
    // HIP compile time (Debug): 2:05:00 (125 min)
    // HIP compile time (Release): 00:01:21 (1.3 min)
    constexpr type_list<v_type<double, Precision::Pointer>,
                        v_type<double, Precision::AccessorKeep>,
                        v_type<double, Precision::AccessorReduced>,
                        v_type<float, Precision::Pointer>,
                        v_type<float, Precision::AccessorKeep>,
                        v_type<int, Precision::Pointer>,
                        v_type<int, Precision::AccessorReduced>>
        type_list;
    constexpr val_list<i_type, 4> outer_list;
    constexpr val_list<i_type, 8> inner_list;
    constexpr val_list<i_type, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       15, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 512>
        compute_list;
    /*
    // HIP compile time (Debug): 25:00
    // HIP compile time (Release): 00:08 (8s)
    constexpr type_list<v_type<double, Precision::Pointer>,
                        v_type<double, Precision::AccessorKeep>,
                        v_type<double, Precision::AccessorReduced>,
                        v_type<float, Precision::Pointer>,
                        v_type<float, Precision::AccessorKeep>>
        type_list;
    constexpr val_list<i_type, 4> outer_list;
    constexpr val_list<i_type, 8> inner_list;
    constexpr val_list<i_type, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>
        compute_list;
    //*/
    // This is a very small list for debugging purposes
    constexpr type_list<v_type<double, Precision::Pointer>,
                        v_type<double, Precision::AccessorKeep>,
                        v_type<float, Precision::Pointer>,
                        v_type<float, Precision::AccessorKeep>,
                        v_type<double, Precision::AccessorReduced> >
        type_list;
    constexpr val_list<i_type, 256> outer_list;
    constexpr val_list<i_type, 8> inner_list;
    constexpr val_list<i_type, 0, 1, 2, 4, 16, 32, 64, 96, 128> compute_list;
    //*/

    std::random_device r_device;
    std::default_random_engine engine(r_device());
    std::uniform_int_distribution<std::int32_t> dist(1, 1000);
    // std::int32_t  rand_val{dist(engine)};
    std::int8_t rand_val{0};

    /*
    std::cout << "num_elems = " << num_elems << "; Array is "
              << (USE_ARRAY ? "used" : "NOT used") << "; " << '\n';
    */

    {
        // MUST be the largest type of all `type_list` types
        memory<double> data(num_elems);
        data.memset(rand_val);

        // Warmup
        run_benchmark<double, 1, 2, 1>(num_elems, static_cast<double>(rand_val),
                                       data.get());
    }

    print_header();
    synchronize();
    run_benchmark_variations(type_list, outer_list, inner_list, compute_list,
                             num_elems, rand_val);
}
