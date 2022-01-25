#ifndef EXEC_CUDA_CUH_
#define EXEC_CUDA_CUH_

#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>


#include <curand_kernel.h>


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

    void reset()
    {
        CUDA_CALL(cudaEventDestroy(ev_));
        CUDA_CALL(cudaEventCreate(&ev_));
    }

    cudaEvent_t& get() { return ev_; }

private:
    cudaEvent_t ev_;
};


class timer {
public:
    void start() { CUDA_CALL(cudaEventRecord(start_.get(), 0)); }

    void stop()
    {
        CUDA_CALL(cudaEventRecord(end_.get(), 0));
        CUDA_CALL(cudaEventSynchronize(end_.get()));
    }

    void reset()
    {
        start_.reset();
        end_.reset();
    }

    // Returns the time in ms
    double get_time()
    {
        float time{};
        CUDA_CALL(cudaEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

private:
    cuda_event start_;
    cuda_event end_;
};


template <typename T>
class Memory {
public:
    Memory(std::size_t num_elems)
        : data_{nullptr}, num_elems_{num_elems}, size_(num_elems_ * sizeof(T))
    {
        cudaSetDevice(0);
        if (size_ > 0) {
            CUDA_CALL(cudaMalloc(&data_, size_));
        }
    }

    ~Memory() { cudaFree(data_); }

    void re_allocate()
    {
        CUDA_CALL(cudaFree(data_));
        CUDA_CALL(cudaMalloc(&data_, size_));
    }

    void re_allocate(std::size_t new_num_elements)
    {
        CUDA_CALL(cudaFree(data_));
        num_elems_ = new_num_elements;
        size_ = num_elems_ * sizeof(T);
        data_ = nullptr;
        if (size_ > 0) {
            CUDA_CALL(cudaMalloc(&data_, size_));
        }
    }


    T* get() { return data_; }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int8_t val) { CUDA_CALL(cudaMemset(data_, val, size_)); }

    std::vector<T> get_vector() const
    {
        std::vector<T> vec(num_elems_);
        CUDA_CALL(cudaMemcpy(vec.data(), data_, num_elems_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
        return vec;
    }

    template <typename NewType>
    std::vector<NewType> get_reinterpret_vector() const
    {
        static_assert(
            sizeof(NewType) <= sizeof(T),
            "Reinterpret is only safe for smaller or same-size types!");
        std::vector<NewType> vec(num_elems_);
        CUDA_CALL(cudaMemcpy(vec.data(), reinterpret_cast<NewType*>(data_),
                             num_elems_ * sizeof(T), cudaMemcpyDeviceToHost));
        return vec;
    }

private:
    T* data_;
    std::size_t num_elems_;
    std::size_t size_;
};


struct memory {
private:
    using big_type = double;

public:
    static constexpr std::size_t max_elem_size{sizeof(big_type)};

    memory(std::size_t num_elems) : data_{num_elems} {}

    void re_allocate() { data_.re_allocate(); }

    template <typename T>
    T* get()
    {
        static_assert(sizeof(T) <= max_elem_size,
                      "The type you chose is too big!");
        return reinterpret_cast<T*>(data_.get());
    }

    std::size_t get_num_elems() const { return data_.get_num_elems(); }

    std::size_t get_byte_size() const { return data_.get_byte_size(); }

    void memset(std::int8_t val) { data_.memset(val); }

    template <typename T>
    std::vector<T> get_vector() const
    {
        return data_.template get_reinterpret_vector<T>();
    }

private:
    Memory<big_type> data_;
};


class RandomNumberGenerator {
public:
    RandomNumberGenerator() : curand_{0} {}

    curandState_t *get_memory() { return curand_.get(); }

    void re_allocate(std::size_t num_elements)
    {
        curand_.re_allocate(num_elements);
    }

    // Returns true if a re-allocation was performed.
    // If true is returned, the data needs to be re-initialized
    bool prepare_for(std::size_t num_elements)
    {
        if (curand_.get_num_elems() < num_elements) {
            this->re_allocate(num_elements);
            return true;
        }
        return false;
    }

    std::size_t get_num_elems() const { return curand_.get_num_elems(); }

private:
    Memory<curandState_t> curand_;
};


#endif  // EXEC_CUDA_CUH_
