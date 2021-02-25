#ifndef EXEC_HELPER_HPP_
#define EXEC_HELPER_HPP_

#include <cstddef>

#include "config.hpp"

constexpr int ceildiv(std::size_t dividend, std::size_t divisor) {
    return (dividend - 1) / divisor + 1;
}

/**Defined in this header:

void synchronize();


struct memory {
   public:
    memory(std::size_t num_elems);
    ~memory();
    T *get();
    std::size_t get_num_elems() const;
    std::size_t get_byte_size() const;
    void memset(std::int8_t val);
    std::vector<T> get_vector() const;
};


class timer {
   public:
    void start();
    void stop();
    void reset();
    // Returns the time in ms
    double get_time();
};
*/

#if ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_CUDA
#include "cuda/exec_cuda.cuh"
#elif ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_HIP
#include "hip/exec_hip.hip.hpp"
#elif ROOFLINE_ARCHITECTURE == ROOFLINE_ARCHITECTURE_CPU
#include "cpu/exec_cpu.hpp"
#endif

#endif  // EXEC_HELPER_HPP_
