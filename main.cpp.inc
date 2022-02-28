#include <algorithm>
#include <cinttypes>
#include <cstdlib>  // for std::exit
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


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
    using arithmetic_type = T;
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
    using arithmetic_type = T;

    if (curr < end && *curr == compute_k) {
        auto info =
            run_benchmark<arithmetic_type, outer, inner, compute_k>(args...);
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


template <typename T>
struct type_to_char {};

template <>
struct type_to_char<double> {
    static constexpr char c = 'd';
};

template <>
struct type_to_char<float> {
    static constexpr char c = 'f';
};

template <>
struct type_to_char<int> {
    static constexpr char c = 'i';
};


template <typename Iterator, typename... Args>
void run_benchmark_variations(Iterator curr, Iterator end, Args&&... args)
{
    if (curr == end) {
        return;
    } else if (curr->first == type_to_char<double>::c) {
        run_benchmark_variations_outer<double>(args..., curr->second);
    } else if (curr->first == type_to_char<float>::c) {
        run_benchmark_variations_outer<float>(args..., curr->second);
    } else if (curr->first == type_to_char<int>::c) {
        run_benchmark_variations_outer<int>(args..., curr->second);
    } else {
        throw std::runtime_error(std::string("Unsupported arithmetic_type ") +
                                 curr->first);
    }
    run_benchmark_variations(curr + 1, end, std::forward<Args>(args)...);
}


template <typename IT, typename... Args>
std::vector<IT> to_vector(val_list<IT>, Args&&... args)
{
    return {std::forward<Args>(args)...};
}

template <typename IT, IT curr_val, IT... RemVals, typename... Args>
std::vector<IT> to_vector(val_list<IT, curr_val, RemVals...>, Args&&... args)
{
    return to_vector(val_list<IT, RemVals...>{}, std::forward<Args>(args)...,
                     curr_val);
}


Precision str_to_prec(const std::string& str)
{
    if (str == "pt") {
        return Precision::Pointer;
    } else if (str == "ak") {
        return Precision::AccessorKeep;
    } else if (str == "ar") {
        return Precision::AccessorReduced;
    } else if (str == "ap") {
        return Precision::AccessorPosit;
    } else {
        throw std::runtime_error("Given string is not supported");
    }
}


std::string prec_to_str(const Precision p)
{
    if (p == Precision::Pointer) {
        return "pt";
    } else if (p == Precision::AccessorKeep) {
        return "ak";
    } else if (p == Precision::AccessorReduced) {
        return "ar";
    } else if (p == Precision::AccessorPosit) {
        return "ap";
    } else {
        throw std::runtime_error("Given string is not supported");
    }
}


template <typename... Args>
std::vector<std::pair<char, Precision>> to_vector(type_list<>, Args&&... args)
{
    return {std::forward<Args>(args)...};
}

template <typename TypePair, typename... RemTypes, typename... Args>
std::vector<std::pair<char, Precision>> to_vector(
    type_list<TypePair, RemTypes...>, Args&&... args)
{
    return to_vector(
        type_list<RemTypes...>{}, std::forward<Args>(args)...,
        std::make_pair(type_to_char<typename TypePair::arithmetic_type>::c,
                       TypePair::p));
}


int main(int argc, char* argv[])
{
    using i_type = std::int32_t;

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
        default_type_list;

    // All of the following MUST be sorted!
    constexpr val_list<i_type, 1, 4> supported_outer_list;
    constexpr val_list<i_type, 2, 8, 16> supported_inner_list;
    constexpr val_list<i_type, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       15, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 512>
        supported_compute_list;

    int number_runs = 20;

    // Values mut be sorted!
    auto type_list = to_vector(default_type_list);
    auto outer_list = to_vector(supported_outer_list);
    auto inner_list = to_vector(supported_inner_list);
    auto compute_list = to_vector(supported_compute_list);


    // I/O conversions and tests
    std::vector<std::string> input_args(argc - 1);
    for (int i = 1; i < argc; ++i) {
        input_args[i - 1] = argv[i];
    }
    const auto begin = input_args.begin();
    const auto end = input_args.end();

    auto get_default_i_list_string = [](const std::vector<i_type>& vec) {
        std::stringstream ss;
        for (std::size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i];
            if (i != vec.size() - 1) {
                ss << ',';
            }
        }
        return ss.str();
    };
    auto get_default_type_list_string =
        [](const std::vector<std::pair<char, Precision>>& vec) {
            std::stringstream ss;
            for (std::size_t i = 0; i < vec.size(); ++i) {
                ss << vec[i].first << '-' << prec_to_str(vec[i].second);
                if (i != vec.size() - 1) {
                    ss << ',';
                }
            }
            return ss.str();
        };
    auto print_help = [&]() {
        std::cout << "Usage: " << argv[0] << "[OPTION]" << '\n';
        std::cout << "Supported options are:\n";
        std::cout << "    --num_elems NUM_ELEMS  Specify the number of "
                     "elements that are used in this benchmark\n"
                  << "        Default value: " << num_elems << '\n';
        std::cout << "    --num_runs NUM_RUNS  Specify the number of "
                     "runs that each kernel will be executed\n"
                  << "        Default value: " << number_runs << '\n';
        std::cout
            << "    --inner LIST_INNER  Specify the different inner template "
               "parameter values with a comma separated list\n"
            << "        Default value: "
            << get_default_i_list_string(inner_list) << '\n';
        std::cout
            << "    --outer LIST_OUTER  Specify the different outer template "
               "parameter values with a comma separated list\n"
            << "        Default value: "
            << get_default_i_list_string(outer_list) << '\n';
        std::cout << "    --compute LIST_COMPUTE  Specify the different "
                     "compute template "
                     "parameter values with a comma separated list\n"
                  << "        Default value: "
                  << get_default_i_list_string(compute_list) << '\n';
        std::cout << "    --type LIST_TYPE  Specify the different "
                     "type combinations AR-ST containing the ARithmetic type "
                     "and the STorage type with a comma separated list\n"
                  << "        Default value: "
                  << get_default_type_list_string(type_list) << '\n';
    };

    auto convert_to_i_vector = [](const std::string& str) {
        std::vector<i_type> result;
        std::stringstream ss(str);
        for (long long val; ss >> val;) {
            result.push_back(val);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        /* Slower alternative
        while (ss.good()) {
            std::string substr;
            std::getline(ss, substr, ',');
            result.push_back(std::stoll(substr));
        }
        */
        return result;
    };

    auto convert_to_type_vector = [](const std::string& str) {
        std::vector<std::pair<char, Precision>> result;
        std::stringstream ss(str);
        while (ss.good()) {
            std::string substr;
            std::getline(ss, substr, ',');
            if (substr[1] != '-' || substr.size() != 4) {
                throw std::runtime_error("Unable to parse type value: " +
                                         substr);
            }
            char type = substr[0];
            Precision p = str_to_prec(substr.substr(2));
            result.push_back(std::make_pair(type, p));
        }
        return result;
    };

    auto pos = begin;
    if (std::find(begin, end, std::string("--help")) != end) {
        print_help();
        return 0;
    }
    if ((pos = std::find(begin, end, std::string("--num_elems"))) != end) {
        num_elems = std::stoll(*(pos + 1));
    }
    if ((pos = std::find(begin, end, std::string("--num_runs"))) != end) {
        number_runs = std::stoll(*(pos + 1));
    }
    if ((pos = std::find(begin, end, std::string("--inner"))) != end) {
        inner_list = convert_to_i_vector(*(pos + 1));
    }
    if ((pos = std::find(begin, end, std::string("--outer"))) != end) {
        outer_list = convert_to_i_vector(*(pos + 1));
    }
    if ((pos = std::find(begin, end, std::string("--compute"))) != end) {
        compute_list = convert_to_i_vector(*(pos + 1));
    }
    if ((pos = std::find(begin, end, std::string("--type"))) != end) {
        type_list = convert_to_type_vector(*(pos + 1));
    }
    /*
     * For the actual type list: <AR-ST>
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

    const unsigned seed{42};

    /*
    std::cout << "num_elems = " << num_elems << "; Array is "
              << (USE_ARRAY ? "used" : "NOT used") << "; " << '\n';
    */

    // Sorting must be ensured for the translation runtime -> compile time
    std::sort(type_list.begin(), type_list.end());
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
        type_list.begin(), type_list.end(), supported_outer_list,
        outer_list.begin(), outer_list.end(), supported_inner_list,
        inner_list.begin(), inner_list.end(), supported_compute_list,
        compute_list.begin(), compute_list.end(), number_runs, num_elems, data,
        seed, rng);
}
