#include <cinttypes>
#include <random>

#include "benchmark.cuh"
#include "helper.cuh"

int main() {
    using value_type = double;
    constexpr std::size_t num_elems = 32 * 1024 * 1024;
    constexpr std::int32_t block_size = 256;
    constexpr std::int32_t fusion_degree = 4;
    constexpr std::int32_t granularity = 8;
    constexpr std::int32_t compute_iter{128};

    std::random_device r_device;
    std::default_random_engine engine(r_device());
    std::uniform_int_distribution<std::int32_t> dist(1, 1000);
    // value_type rand_val{static_cast<value_type>(dist(engine))};
    value_type rand_val{static_cast<value_type>(1)};
    cudaSetDevice(0);

    std::cout << "num_elems = " << num_elems
              << "; fusion_degree = " << fusion_degree
              << "; granularity = " << granularity
              << "; compute_iter = " << compute_iter << '\n';

    c_memory<value_type> data(num_elems);
    data.memset(0);
    benchmark<value_type, block_size, granularity, fusion_degree, compute_iter>
        warmup(num_elems);

    {
        // Warmup
        for (int i = 0; i < 30; ++i) {
            warmup.run(rand_val, data.get());
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());

    {
        benchmark<double, block_size, granularity, fusion_degree, compute_iter>
            bench(num_elems);
        auto time = bench.run(rand_val, data.get());
        std::cout << "double: " << bench.get_bandwidth() << " GB/s; "
                  << bench.get_flops() << " GFLOPs; " << time << " ms\n";
    }
    {
        benchmark<float, block_size, granularity, fusion_degree, compute_iter>
            bench(num_elems);
        auto time = bench.run(static_cast<float>(rand_val),
                  reinterpret_cast<float *>(data.get()));
        std::cout << "float:  " << bench.get_bandwidth() << " GB/s; "
                  << bench.get_flops() << " GFLOPs; " << time << " ms\n";
    }
    {
        benchmark<int, block_size, granularity, fusion_degree, compute_iter>
            bench(num_elems);
        auto time = bench.run(static_cast<int>(rand_val),
                  reinterpret_cast<int *>(data.get()));
        std::cout << "int:    " << bench.get_bandwidth() << " GB/s; "
                  << bench.get_flops() << " GFLOPs; " << time << " ms\n";
    }
}
