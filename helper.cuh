#pragma once

#include <vector>
#include <iostream>


#define CUDA_CALL(call)                                                  \
    do {                                                                 \
        auto err = call;                                                 \
        if (err) {                                                       \
            std::cerr << "Cuda error in file " << __FILE__               \
                      << " L:" << __LINE__                               \
                      << "; Error: " << cudaGetErrorString(err) << '\n'; \
        }                                                                \
    } while (false)


struct cuda_event {
   public:
    cuda_event() { CUDA_CALL(cudaEventCreate(&ev_)); }

    ~cuda_event() { CUDA_CALL(cudaEventDestroy(ev_)); }

    void reset() {
        CUDA_CALL(cudaEventDestroy(ev_));
        CUDA_CALL(cudaEventCreate(&ev_));
    }

    cudaEvent_t &get() { return ev_; }

   private:
    cudaEvent_t ev_;
};

class cuda_timer {

   public:
    void start() {
        CUDA_CALL(cudaEventRecord(start_.get(), 0));
        // ch_start_ = std::chrono::steady_clock::now();
    }

    void stop() {
        CUDA_CALL(cudaEventRecord(end_.get(), 0));
        CUDA_CALL(cudaEventSynchronize(end_.get()));
        // ch_end_ = std::chrono::steady_clock::now();
    }

    void reset() {
        start_.reset();
        end_.reset();
    }

    // Returns the time in ms
    double get_time() {
        float time{};
        CUDA_CALL(cudaEventElapsedTime(&time, start_.get(), end_.get()));
        // std::chrono::duration<double> ch_time = ch_end_ - ch_start_;
        return time;
    }

   private:
    cuda_event start_;
    cuda_event end_;
    //using time_point = decltype(std::chrono::steady_clock::now());
    //time_point ch_start_;
    //time_point ch_end_;
};

template <typename T>
struct c_memory {
   public:
    c_memory(std::size_t num_elems)
        : num_elems_(num_elems), size_(num_elems_ * sizeof(T)) {
        CUDA_CALL(cudaMalloc(&data_, size_));
    }

    ~c_memory() { CUDA_CALL(cudaFree(data_)); }

    T *get() { return data_; }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int32_t val) {
        CUDA_CALL(cudaMemset(data_, val, size_ / sizeof(std::int32_t)));
    }

    std::vector<T> get_vector() {
        std::vector<T> vec(num_elems_);
        CUDA_CALL(cudaMemcpy(vec.data(), data_, size_, cudaMemcpyDeviceToHost));
        return vec;
    }

   private:
    T *data_;
    std::size_t num_elems_;
    std::size_t size_;
};

constexpr int ceildiv(std::size_t dividend, std::size_t divisor) {
    return (dividend - 1) / divisor + 1;
}


