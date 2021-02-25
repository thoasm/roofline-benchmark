// Required includes:
// cinttypes
// exec_helper.hpp for ceildiv

/**
 * Note: when changing the `input` type to int, followed by casting it to `T` /
 * `value_type`, both the Accessor and the Pointer implementation are
 * significantly faster for double computations 128 and 256 compared to keeping
 * `T input`.
 * Currently, it is unclear why that happens!
 */

template <std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters, typename T>
__global__ void benchmark_kernel(const T input, T *__restrict__ data) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(outer_work_iters > 0, "outer_work_iters must be positive!");
    static_assert(compute_iters >= 0,
                  "compute_iters must be positive or zero!");
    /*
    const std::int32_t idx =
        blockIdx.x * block_size * inner_work_iters + threadIdx.x;
    const std::int32_t inner_stride = block_size;
    const std::int32_t outer_stride = gridDim.x * block_size * inner_work_iters;
    /*/
    const std::int64_t idx = blockIdx.x * block_size + threadIdx.x;
    const std::int64_t inner_stride = gridDim.x * block_size;
    const std::int64_t outer_stride = inner_work_iters * inner_stride;
    //*/
    // const T input = static_cast<T>(i_input);

#if USE_ARRAY
    static_assert(inner_work_iters % 2 == 0,
                  "inner_work_iters must be dividable by 2!");
    T reg[inner_work_iters];
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            reg[i] = data[idx + i * inner_stride + o * outer_stride];
#pragma unroll 128
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg[i] = reg[i] * reg[i] + input;
            }
        }
        T reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<T>(-1)) {
            data[idx + o * outer_stride] = reduced;
        }
    }

#else
    T reg{1};
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            const T mem = data[idx + i * inner_stride + o * outer_stride];
            reg = mem * input + reg;
#pragma unroll 128
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg = reg * reg + input;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<T>(-1)) {
            data[idx + o * outer_stride] = reg;
        }
    }
#endif  // USE_ARRAY
}

// Specialization for Accessor
template <std::int32_t block_size, std::int32_t outer_work_iters,
          std::int32_t inner_work_iters, std::int32_t compute_iters,
          typename Input, typename Accessor>
__global__ void benchmark_accessor_kernel(const Input input, Accessor acc) {
    static_assert(block_size > 0, "block_size must be positive!");
    static_assert(outer_work_iters > 0, "outer_work_iters must be positive!");
    static_assert(compute_iters >= 0,
                  "compute_iters must be positive or zero!");
    static_assert(Accessor::dimensionality == 3, "Accessor must be 3D!");
    /*
    const std::int32_t idx =
        blockIdx.x * block_size * inner_work_iters + threadIdx.x;
    const std::int32_t outer_stride = gridDim.x * block_size * inner_work_iters;
    /*/
    const std::int64_t idx = blockIdx.x * block_size + threadIdx.x;

    using value_type = typename Accessor::accessor::arithmetic_type;
    static_assert(std::is_same<value_type, Input>::value, "Types must match!");
    //*/
    // const auto input = static_cast<value_type>(i_input);

#if USE_ARRAY
    static_assert(inner_work_iters % 2 == 0,
                  "inner_work_iters must be dividable by 2!");
    value_type reg[inner_work_iters];
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            reg[i] = acc(o, i, idx);
            // reg[i] = acc(idx + i * inner_stride + o * outer_stride);
#pragma unroll 128
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg[i] = reg[i] * reg[i] + input;
            }
        }
        value_type reduced{};
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; i += 2) {
            reduced = reg[i] * reg[i + 1] + reduced;
        }
        // Intentionally is never true
        if (reduced == static_cast<value_type>(-1)) {
            acc(o, 0, idx) = reduced;
            // acc(idx + o * outer_stride) = reduced;
        }
    }

#else
    value_type reg{1};
    for (std::int32_t o = 0; o < outer_work_iters; ++o) {
#pragma unroll
        for (std::int32_t i = 0; i < inner_work_iters; ++i) {
            // const value_type mem =
            //    acc(idx + i * inner_stride + o * outer_stride);
            const value_type mem = acc(o, i, idx);
            reg = mem * input + reg;
#pragma unroll 128
            for (std::int32_t c = 0; c < compute_iters; ++c) {
                reg = reg * reg + input;
            }
        }
        // Intentionally is never true
        if (reg == static_cast<value_type>(-1)) {
            acc(o, 0, idx) = reg;
            // acc(idx + o * outer_stride) = reg;
        }
    }
#endif  // USE_ARRAY
}

