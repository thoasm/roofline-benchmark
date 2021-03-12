#ifndef EXEC_CUDA_CUH_
#define EXEC_CUDA_CUH_

#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CUDA_CALL(call)                                                  \
    do {                                                                 \
        auto err = call;                                                 \
        if (err != cudaSuccess) {                                        \
            std::cerr << "Cuda error in file " << __FILE__               \
                      << " L:" << __LINE__                               \
                      << "; Error: " << cudaGetErrorString(err) << '\n'; \
            throw std::runtime_error(cudaGetErrorString(err));           \
        }                                                                \
    } while (false)

void synchronize() { CUDA_CALL(cudaDeviceSynchronize()); }

struct cuda_event {
   public:
    cuda_event() { CUDA_CALL(cudaEventCreate(&ev_)); }

    ~cuda_event() { cudaEventDestroy(ev_); }

    void reset() {
        CUDA_CALL(cudaEventDestroy(ev_));
        CUDA_CALL(cudaEventCreate(&ev_));
    }

    cudaEvent_t &get() { return ev_; }

   private:
    cudaEvent_t ev_;
};

class timer {
   public:
    void start() { CUDA_CALL(cudaEventRecord(start_.get(), 0)); }

    void stop() {
        CUDA_CALL(cudaEventRecord(end_.get(), 0));
        CUDA_CALL(cudaEventSynchronize(end_.get()));
    }

    void reset() {
        start_.reset();
        end_.reset();
    }

    // Returns the time in ms
    double get_time() {
        float time{};
        CUDA_CALL(cudaEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

   private:
    cuda_event start_;
    cuda_event end_;
};

struct memory {
   private:
    using big_type = double;

   public:
    static constexpr std::size_t max_elem_size{sizeof(big_type)};

    memory(std::size_t num_elems)
        : num_elems_{num_elems}, size_(num_elems_ * max_elem_size) {
        cudaSetDevice(0);
        CUDA_CALL(cudaMalloc(&data_, size_));
    }

    ~memory() { cudaFree(data_); }

    void re_allocate() {
        CUDA_CALL(cudaFree(data_));
        CUDA_CALL(cudaMalloc(&data_, size_));
    }

    template <typename T>
    T *get() {
        static_assert(sizeof(T) <= max_elem_size,
                      "The type you chose is too big!");
        return reinterpret_cast<T *>(data_);
    }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int8_t val) { CUDA_CALL(cudaMemset(data_, val, size_)); }

    template <typename T>
    std::vector<T> get_vector() const {
        std::vector<T> vec(num_elems_);
        CUDA_CALL(cudaMemcpy(vec.data(), data_, num_elems_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
        return vec;
    }

   private:
    void *data_;
    std::size_t num_elems_;
    std::size_t size_;
};

#endif  // EXEC_CUDA_CUH_
