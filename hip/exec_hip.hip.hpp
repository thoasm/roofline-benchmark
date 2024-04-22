#ifndef EXEC_HIP_HIP_HPP_
#define EXEC_HIP_HIP_HPP_

#include <hip/hip_runtime.h>


#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <hiprand/hiprand_kernel.h>


#define HIP_CALL(call)                                                         \
    do {                                                                       \
        auto err = call;                                                       \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error in file " << __FILE__ << " L:" << __LINE__ \
                      << "; Error: " << err << '\n';                           \
            throw std::runtime_error("Bad HIP call!");                         \
        }                                                                      \
    } while (false)


void synchronize() { HIP_CALL(hipDeviceSynchronize()); }


struct hip_event {
public:
    hip_event() { HIP_CALL(hipEventCreate(&ev_)); }

    ~hip_event() { hipEventDestroy(ev_); }

    void reset()
    {
        HIP_CALL(hipEventDestroy(ev_));
        HIP_CALL(hipEventCreate(&ev_));
    }

    hipEvent_t& get() { return ev_; }

private:
    hipEvent_t ev_;
};


class timer {
public:
    void start() { HIP_CALL(hipEventRecord(start_.get(), 0)); }

    void stop()
    {
        HIP_CALL(hipEventRecord(end_.get(), 0));
        HIP_CALL(hipEventSynchronize(end_.get()));
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
        HIP_CALL(hipEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

private:
    hip_event start_;
    hip_event end_;
};


template <typename T>
class Memory {
public:
    Memory(std::size_t num_elems)
        : data_{nullptr}, num_elems_{num_elems}, size_{num_elems_ * sizeof(T)}
    {
        hipSetDevice(0);
        if (size_ > 0) {
            HIP_CALL(hipMalloc(&data_, size_));
        }
    }

    ~Memory() { hipFree(data_); }

    void re_allocate()
    {
        HIP_CALL(hipFree(data_));
        HIP_CALL(hipMalloc(&data_, size_));
    }

    void re_allocate(std::size_t new_num_elements)
    {
        HIP_CALL(hipFree(data_));
        num_elems_ = new_num_elements;
        size_ = num_elems_ * sizeof(T);
        data_ = nullptr;
        if (size_ > 0) {
            HIP_CALL(hipMalloc(&data_, size_));
        }
    }


    T* get() { return data_; }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int8_t val) { HIP_CALL(hipMemset(data_, val, size_)); }

    std::vector<T> get_vector() const
    {
        std::vector<T> vec(num_elems_);
        HIP_CALL(hipMemcpy(vec.data(), data_, num_elems_ * sizeof(T),
                           hipMemcpyDeviceToHost));
        return vec;
    }

    template <typename NewType>
    std::vector<NewType> get_reinterpret_vector() const
    {
        static_assert(
            sizeof(NewType) <= sizeof(T),
            "Reinterpret is only safe for smaller or same-size types!");
        std::vector<NewType> vec(num_elems_);
        HIP_CALL(hipMemcpy(vec.data(), reinterpret_cast<NewType*>(data_),
                           num_elems_ * sizeof(T), hipMemcpyDeviceToHost));
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
    RandomNumberGenerator() : hiprand_{0} {}

    hiprandState_t* get_memory() { return hiprand_.get(); }

    void re_allocate(std::size_t num_elements)
    {
        hiprand_.re_allocate(num_elements);
    }

    // Returns true if a re-allocation was performed.
    // If true is returned, the data needs to be re-initialized
    bool prepare_for(std::size_t num_elements)
    {
        if (hiprand_.get_num_elems() < num_elements) {
            this->re_allocate(num_elements);
            return true;
        }
        return false;
    }

    std::size_t get_num_elems() const { return hiprand_.get_num_elems(); }

private:
    Memory<hiprandState_t> hiprand_;
};


#endif  // EXEC_HIP_HIP_HPP_
