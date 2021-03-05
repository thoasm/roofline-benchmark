#ifndef EXEC_CPU_HPP_
#define EXEC_CPU_HPP_

#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <vector>

void synchronize() {}

class timer {
   public:
    void start() {
        synchronize();
        ch_start_ = std::chrono::steady_clock::now();
    }

    void stop() {
        synchronize();
        ch_end_ = std::chrono::steady_clock::now();
    }

    void reset() {
        ch_start_ = time_point{};
        ch_end_ = time_point{};
    }

    // Returns the time in ms
    double get_time() {
        std::chrono::duration<double> ch_time = ch_end_ - ch_start_;
        return ch_time.count() * 1e3;
    }

   private:
    using time_point = decltype(std::chrono::steady_clock::now());
    time_point ch_start_;
    time_point ch_end_;
};

template <typename T>
struct memory {
   public:
    memory(std::size_t num_elems)
        : data_(num_elems),
          num_elems_(num_elems),
          size_(num_elems_ * sizeof(T)) {}

    ~memory() = default;

    T *get() { return data_.data(); }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    void memset(std::int8_t val) {
        constexpr std::size_t chunk_size{4096};
        auto ptr = reinterpret_cast<std::int8_t *>(data_.data());
        //#pragma parallel for schedule(static, 1)
        for (std::size_t i = 0; i < size_; i += chunk_size) {
            std::memset(ptr + i, val, chunk_size);
        }
    }

    // Note: copies the vector here to have the same principle as the others
    std::vector<T> get_vector() const { return data_; }

   private:
    std::vector<T> data_;
    std::size_t num_elems_;
    std::size_t size_;
};

#endif  //  EXEC_CPU_HPP_
