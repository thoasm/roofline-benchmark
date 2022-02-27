#include <algorithm>
#include <cinttypes>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cstdlib>  // for std::exit


#include "benchmark.hpp"
#include "benchmark_info.hpp"
#include "exec_helper.hpp"


constexpr char SEP = ';';
constexpr char NL = '\n';


void print_header()
{
    // clang-format off
    std::cout << std::right;
    std::cout << std::setw(15) << "Precision"
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


void print_info(const benchmark_info& info)
{
    // clang-format off
    // std::cout << std::defaultfloat << std::setprecision(5);
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::right;
    std::cout << std::setw(15) << info.precision
              << SEP << std::setw(11) << info.get_giops()
              << SEP << std::setw(11) << info.get_bw_gbs()
              << SEP << std::setw(11) << info.time_ms
              << SEP << std::setw(13) << info.computations
              << SEP << std::setw(13) << info.memory_moved_bytes
              << SEP << std::setw(10) << info.outer_work_iters
              << SEP << std::setw(10) << info.inner_work_iters
              << SEP << std::setw(9) << info.compute_iters
              << SEP << std::setw(11) << info.num_elems
              << NL << std::flush;
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
template <typename T, typename IT, IT outer, IT inner, typename Iterator,
          typename... Args>
void run_benchmark_variations_compute(val_list<IT>, Iterator curr, Iterator end,
                                      Args&&... args)
{
    if (curr != end) {
        std::cerr << "Unsupported compute value: " << *curr << '\n';
        std::exit(1);
    }
}

template <typename T, typename IT, IT outer, IT inner, IT compute_k,
          IT... rem_compute, typename Iterator, typename... Args>
void run_benchmark_variations_compute(val_list<IT, compute_k, rem_compute...>,
                                      Iterator curr, Iterator end,
                                      Args&&... args)
{
    using value_type = typename T::type;
    auto prec = T::p;

    if (curr < end && *curr == compute_k) {
        auto info =
            run_benchmark<value_type, outer, inner, compute_k>(args..., prec);
        print_info(info);
        ++curr;
    }

    // recursion
    run_benchmark_variations_compute<T, IT, outer, inner>(
        val_list<IT, rem_compute...>{}, curr, end, std::forward<Args>(args)...);
}


// For inner resolution
template <typename T, typename IT, IT outer, typename Iterator,
          typename... Args>
void run_benchmark_variations_inner(val_list<IT>, Iterator curr, Iterator end,
                                    Args&&... args)
{
    if (curr != end) {
        std::cerr << "Unsupported inner value: " << *curr << '\n';
        std::exit(1);
    }
}

template <typename T, typename IT, IT outer, IT inner_k, IT... rem_inner,
          typename Iterator, typename... Args>
void run_benchmark_variations_inner(val_list<IT, inner_k, rem_inner...>,
                                    Iterator curr, Iterator end, Args&&... args)
{
    if (curr < end && *curr == inner_k) {
        run_benchmark_variations_compute<T, IT, outer, inner_k>(
            std::forward<Args>(args)...);
        ++curr;
    }
    // recursion
    run_benchmark_variations_inner<T, IT, outer>(
        val_list<IT, rem_inner...>{}, curr, end, std::forward<Args>(args)...);
}


// For outer resolution
template <typename T, typename IT, typename Iterator, typename... Args>
void run_benchmark_variations_outer(val_list<IT>, Iterator curr, Iterator end,
                                    Args&&...)
{
    if (curr != end) {
        std::cerr << "Unsupported outer value: " << *curr << '\n';
        std::exit(1);
    }
}

template <typename T, typename IT, IT outer_k, IT... rem_outer,
          typename Iterator, typename... Args>
void run_benchmark_variations_outer(val_list<IT, outer_k, rem_outer...>,
                                    Iterator curr, Iterator end, Args&&... args)
{
    if (curr < end && *curr == outer_k) {
        run_benchmark_variations_inner<T, IT, outer_k>(
            std::forward<Args>(args)...);
        ++curr;
    }
    // recursion
    run_benchmark_variations_outer<T>(val_list<IT, rem_outer...>{}, curr, end,
                                      std::forward<Args>(args)...);
}


template <typename... Args>
void run_benchmark_variations(type_list<>,  // Iterator curr, Iterator end,
                              Args&&...)
{}

template <typename T, typename... RemTypes, typename... Args>
void run_benchmark_variations(type_list<T, RemTypes...>, Args&&... args)
{
    run_benchmark_variations_outer<T>(std::forward<Args>(args)...);
    // recursion
    run_benchmark_variations(type_list<RemTypes...>{},
                             std::forward<Args>(args)...);
}


template <typename IT, typename... Args>
std::vector<IT> to_vector(val_list<IT>, Args&&... args)
{
    return {std::forward<Args>(args)...};
}

template <typename IT, IT curr_val, IT... RemVals, typename... Args>
std::vector<IT> to_vector(val_list<IT, curr_val, RemVals...>, Args&&... args)
{
    return to_vector(val_list<IT, RemVals...>{}, curr_val,
                     std::forward<Args>(args)...);
}


int main()
{
    using i_type = std::int32_t;

    //*
    std::size_t num_elems = 256 * 1024 * 1024;
    constexpr type_list<v_type<double, Precision::Pointer>,
                        v_type<double, Precision::AccessorKeep>,
                        v_type<double, Precision::AccessorReduced>,
                        v_type<double, Precision::AccessorPosit>,
                        v_type<float, Precision::Pointer>,
                        v_type<float, Precision::AccessorKeep>,
                        v_type<float, Precision::AccessorPosit>,
                        v_type<int, Precision::Pointer>,
                        v_type<int, Precision::AccessorReduced>>
        supported_type_list;

    // All of the following MUST be sorted!
    constexpr val_list<i_type, 4> supported_outer_list;
    constexpr val_list<i_type, 8, 16> supported_inner_list;
    constexpr val_list<i_type, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       15, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 512>
        supported_compute_list;

    int number_runs = 20;

    /*/
    // This is a very small list for debugging purposes
    constexpr std::size_t num_elems = 32 * 1024 * 1024;
    constexpr type_list<v_type<double, Precision::Pointer>,
                        v_type<double, Precision::AccessorKeep>,
                        v_type<double, Precision::AccessorReduced>,
                        v_type<double, Precision::AccessorPosit>,
                        v_type<float, Precision::Pointer>,
                        v_type<float, Precision::AccessorKeep>,
                        v_type<float, Precision::AccessorPosit>>
        type_list;
    constexpr val_list<i_type, 4> outer_list;
    constexpr val_list<i_type, 8> inner_list;
    constexpr val_list<i_type, 0, 1, 32, 128> compute_list;
    //*/
    // Values mut be sorted!

    /*
     * For the actual type list: <AR,ST>
     *
     * AR:
     * d -> double
     * f -> float
     * i -> int
     *
     * ST:
     * pt -> Pointer
     * ak -> AccessorKeep
     * ar -> AccessorReduced
     * ap -> AccessorPosit
     */

    auto outer_list = to_vector(supported_outer_list);
    auto inner_list = to_vector(supported_inner_list);
    auto compute_list = to_vector(supported_compute_list);

    const unsigned seed{42};

    /*
    std::cout << "num_elems = " << num_elems << "; Array is "
              << (USE_ARRAY ? "used" : "NOT used") << "; " << '\n';
    */

    // Sorting must be ensured for the translation runtime -> compile time
    std::sort(outer_list.begin(), outer_list.end());
    std::sort(inner_list.begin(), inner_list.end());
    std::sort(compute_list.begin(), compute_list.end());


    memory data(num_elems);
    RandomNumberGenerator rng;

    // Warmup
    run_benchmark<double, 1, 2, 1>(1, num_elems, data, seed, rng);

    print_header();
    synchronize();
    run_benchmark_variations(
        supported_type_list, supported_outer_list, outer_list.begin(),
        outer_list.end(), supported_inner_list, inner_list.begin(),
        inner_list.end(), short_compute_list, compute_list.begin(),
        compute_list.end(), number_runs, num_elems, data, seed, rng);
}
