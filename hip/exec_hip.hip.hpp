#ifndef EXEC_CUDA_CUH_
#define EXEC_CUDA_CUH_

#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#define HIP_CALL(call)                                                    \
    do {                                                                  \
        auto err = call;                                                  \
        if (err != hipSuccess) {                                          \
            std::cerr << "Cuda error in file " << __FILE__                \
                      << " L:" << __LINE__ << "; Error: " << err << '\n'; \
            throw std::runtime_error("Bad HIP call!");                    \
        }                                                                 \
    } while (false)

void synchronize() { HIP_CALL(hipDeviceSynchronize()); }

struct hip_event {
   public:
    hip_event() { HIP_CALL(hipEventCreate(&ev_)); }

    ~hip_event() { hipEventDestroy(ev_); }

    void reset() {
        HIP_CALL(hipEventDestroy(ev_));
        HIP_CALL(hipEventCreate(&ev_));
    }

    hipEvent_t &get() { return ev_; }

   private:
    hipEvent_t ev_;
};

class timer {
   public:
    void start() { HIP_CALL(hipEventRecord(start_.get(), 0)); }

    void stop() {
        HIP_CALL(hipEventRecord(end_.get(), 0));
        HIP_CALL(hipEventSynchronize(end_.get()));
    }

    void reset() {
        start_.reset();
        end_.reset();
    }

    // Returns the time in ms
    double get_time() {
        float time{};
        HIP_CALL(hipEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

   private:
    hip_event start_;
    hip_event end_;
};

template <typename T>
struct memory {
   public:
    memory(std::size_t num_elems)
        : num_elems_(num_elems), size_(num_elems_ * sizeof(T)) {
        HIP_CALL(hipMalloc(&data_, size_));
    }

    ~memory() { hipFree(data_); }

    T *get() { return data_; }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int8_t val) { HIP_CALL(hipMemset(data_, val, size_)); }

    std::vector<T> get_vector() const {
        std::vector<T> vec(num_elems_);
        HIP_CALL(hipMemcpy(vec.data(), data_, size_, hipMemcpyDeviceToHost));
        return vec;
    }

   private:
    T *data_;
    std::size_t num_elems_;
    std::size_t size_;
};

#endif  // EXEC_CUDA_CUH_
