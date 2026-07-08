#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include </usr/local/cuda/include/cuda_runtime_api.h>
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

#include "ntt_config.h"

struct ModulusRuntime {
    uint32_t qi;
    uint32_t primitive_root;
    uint32_t psi;   // primitive 2N-th root
    uint32_t omega; // psi^2, primitive N-th root
};

#define SEED 1234
#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 30

#define BLOCK_DIM 256

__host__ __device__ inline uint32_t add_mod(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t sum = static_cast<uint64_t>(a) + b;
    if (sum >= q) sum -= q;
    return static_cast<uint32_t>(sum);
}

__host__ __device__ inline uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t q) {
    if (a >= b) return a - b;
    return static_cast<uint32_t>(static_cast<uint64_t>(a) + q - b);
}

__host__ __device__ inline uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t mul = static_cast<uint64_t>(a) * b;
    mul %= q;
    return static_cast<uint32_t>(mul);
}

__device__ inline void butterfly(uint32_t* A_limb, const uint32_t* tw_limb,
                                 size_t t, size_t stage_block,
                                 size_t stage_id, size_t m, uint32_t q) {
    size_t a = stage_block * 2 * t + stage_id;
    uint32_t u = A_limb[a];
    uint32_t v = mul_mod(A_limb[a + t], tw_limb[m + stage_block], q);
    A_limb[a] = add_mod(u, v, q);
    A_limb[a + t] = sub_mod(u, v, q);
}

__global__ void NTTStage(uint32_t* A, const uint32_t* twiddles,
                         const uint32_t* moduli, size_t N,
                         size_t m, size_t t) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (N >> 1)) {
        return;
    }

    size_t limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = twiddles + limb * N;

    size_t stage_block = tid / t;
    size_t stage_id = tid % t;
    butterfly(A_limb, tw_limb, t, stage_block, stage_id, m, q);
}

void launch_ntt(uint32_t* A_dev, const uint32_t* twiddle_dev,
                const uint32_t* moduli_dev, size_t N, int logN,
                size_t limb_count, dim3 dimBlock) {
    size_t butterflies = N >> 1;
    unsigned int grid_x =
        static_cast<unsigned int>((butterflies + dimBlock.x - 1) / dimBlock.x);
    dim3 dimGrid(grid_x, static_cast<unsigned int>(limb_count), 1);

    size_t t = N >> 1;
    for (int stage = 0; stage < logN; ++stage) {
        size_t m = size_t{1} << stage;
        NTTStage<<<dimGrid, dimBlock>>>(A_dev, twiddle_dev, moduli_dev, N, m, t);
        t >>= 1;
    }
}

uint32_t pow_mod(uint32_t base, uint64_t exp, uint32_t mod) {
    uint64_t result = 1;
    uint64_t x = base;

    while (exp > 0) {
        if (exp & 1) result = (result * x) % mod;
        x = (x * x) % mod;
        exp >>= 1;
    }
    return static_cast<uint32_t>(result);
}

uint32_t bit_reverse(uint32_t value, int width) {
    uint32_t reversed = 0;
    for (int i = 0; i < width; ++i) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

ModulusRuntime make_modulus_runtime(int logN, const ModulusConfig& config) {
    uint64_t N = uint64_t{1} << logN;
    uint64_t root_order = 2 * N;

    if ((static_cast<uint64_t>(config.qi) - 1) % root_order != 0) {
        std::cerr << "qi must satisfy qi == 1 mod 2N for negacyclic NTT.\n";
        std::exit(EXIT_FAILURE);
    }

    uint32_t psi = pow_mod(
        config.primitive_root,
        (static_cast<uint64_t>(config.qi) - 1) / root_order,
        config.qi);
    uint32_t omega = static_cast<uint32_t>(
        (static_cast<uint64_t>(psi) * psi) % config.qi);

    if (pow_mod(psi, root_order, config.qi) != 1 ||
        pow_mod(psi, N, config.qi) != config.qi - 1) {
        std::cerr << "primitive_root did not generate a primitive 2N-th root.\n";
        std::exit(EXIT_FAILURE);
    }

    return {config.qi, config.primitive_root, psi, omega};
}

std::vector<uint32_t> make_twiddle_table(int logN, uint32_t q, uint32_t psi) {
    size_t N = size_t{1} << logN;
    std::vector<uint32_t> twiddles(N);
    for (size_t i = 0; i < N; ++i) {
        twiddles[i] = pow_mod(psi, bit_reverse(static_cast<uint32_t>(i), logN), q);
    }
    return twiddles;
}

void cpu_ntt_direct(uint32_t* a, int logN, uint32_t q, const uint32_t* psi) {
    size_t N = size_t{1} << logN;
    std::vector<uint32_t> out(N);

    for (size_t k = 0; k < N; ++k) {
        uint64_t power = 1;
        uint64_t sum = 0;

        for (size_t j = 0; j < N; ++j) {
            uint64_t term = (static_cast<uint64_t>(a[j]) * power) % q;
            sum += term;
            if (sum >= q) sum -= q;
            power = (power * psi[k]) % q;
        }

        out[k] = static_cast<uint32_t>(sum);
    }

    std::copy(out.begin(), out.end(), a);
}

// bit-reversed order
void cpu_ntt_butterfly(uint32_t* a, int logN, uint32_t q, const uint32_t* psi) {
    size_t N = size_t{1} << logN;
    size_t t = N >> 1;

    for (size_t m = 1; m < N; m *= 2) {
        for (size_t j = 0; j < m; j += 1){
            for (size_t k = j * 2 * t; k < j * 2 * t + t; k += 1){
                uint32_t u = a[k];
                uint32_t v = static_cast<uint32_t>(
                    (static_cast<uint64_t>(a[k + t]) * psi[m + j]) % q);
                a[k] = add_mod(u, v, q);
                a[k + t] = sub_mod(u, v, q);
            }
        }
        t >>= 1;
    }
}

void fill_random(uint32_t* arr, size_t size, uint32_t q) {
    if (q <= 1) {
        std::cerr << "q must be greater than 1 for NTT input generation.\n";
        std::exit(EXIT_FAILURE);
    }

    std::mt19937 gen(SEED);
    std::uniform_int_distribution<uint32_t> dist(0, q - 1);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}

bool validate(const uint32_t* expected, const uint32_t* actual, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (expected[i] != actual[i]) {
            return false;
        }
    }
    return true;
}

double gops(const VersionConfig& config, float elapsed_ms) {
    if (elapsed_ms <= 0.0f) {
        return 0.0;
    }

    double N = (double)(size_t{1} << config.logN);
    double butterflies =
        0.5 * N * (double)config.logN * (double)config.moduli.size();
    double ops = 3.0 * butterflies;
    return ops / ((double)elapsed_ms * 1.0e6);
}

void print_time_result(const char* label, float elapsed_ms,
                       const VersionConfig& config) {
    std::cout << ">>> " << label << " execution time: " << elapsed_ms
              << " ms (" << gops(config, elapsed_ms) << " GOp/s)"
              << std::endl;
}

void print_time_comparison(float custom_ms, float base_ms) {
    if (custom_ms <= 0.0f || base_ms <= 0.0f) return;

    float percent = (base_ms / custom_ms) * 100.0f;
    std::cout << ">>> Custom performance vs Base: "
              << percent
              << "% (Base = 100.0%)" << std::endl;
}

void profiler_range_push(const char* name) {
#ifdef USE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}

void profiler_range_pop() {
#ifdef USE_NVTX
    nvtxRangePop();
#endif
}

void print_first_10(const uint32_t* arr, size_t size) {
    size_t limit = std::min(size, (size_t)10);
    for (size_t i = 0; i < limit; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--verify|-v] [--base|-b] [--help|-h]\n"
              << "  --verify, -v  Copy result back and run CPU validation.\n"
              << "  --base, -b    Compare custom kernel time with base library.\n"
              << "  --help, -h    Show this help message.\n";
}

int main(int argc, char* argv[]) {
    bool verify = false;
    bool compare_base = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify" || arg == "-v") {
            verify = true;
        } else if (arg == "--base" || arg == "-b") {
            compare_base = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    VersionConfig ver_config = default_ntt_config();
    int logN = ver_config.logN;
    size_t N = size_t{1} << logN;
    size_t limb_count = ver_config.moduli.size();
    if (limb_count == 0) {
        std::cerr << "At least one modulus limb is required.\n";
        return 1;
    }

    std::vector<ModulusRuntime> mod_runtime;
    std::vector<uint32_t> moduli_host(limb_count);
    std::vector<uint32_t> twiddle_table(limb_count * N);
    mod_runtime.reserve(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb) {
        ModulusRuntime runtime =
            make_modulus_runtime(logN, ver_config.moduli[limb]);
        mod_runtime.push_back(runtime);
        moduli_host[limb] = runtime.qi;

        std::vector<uint32_t> limb_twiddles =
            make_twiddle_table(logN, runtime.qi, runtime.psi);
        std::copy(limb_twiddles.begin(), limb_twiddles.end(),
                  twiddle_table.begin() + limb * N);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "------------------NTT------------------\n";
    std::cout << "Verification: " << (verify ? "on" : "off") << "\n";
    std::cout << "Base comparison: "
              << (compare_base ? "requested (not implemented)" : "off") << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    std::cout << "Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";
    std::cout << "N: " << N << "\n";
    std::cout << "limbs: " << limb_count << "\n";
    for (size_t limb = 0; limb < limb_count; ++limb) {
        std::cout << "limb " << limb
                  << " qi: " << mod_runtime[limb].qi
                  << ", primitive_root: " << mod_runtime[limb].primitive_root
                  << ", psi: " << mod_runtime[limb].psi
                  << ", omega: " << mod_runtime[limb].omega << "\n";
    }

    size_t total_values = limb_count * N;

    uint32_t* A_host = new uint32_t[total_values];
    uint32_t* A_ref_host = verify ? new uint32_t[total_values] : nullptr;
    uint32_t* out_host = verify ? new uint32_t[total_values] : nullptr;

    for (size_t limb = 0; limb < limb_count; ++limb) {
        fill_random(A_host + limb * N, N, mod_runtime[limb].qi);
    }
    if (verify) {
        std::copy(A_host, A_host + total_values, A_ref_host);
    }

    const size_t bytes = total_values * sizeof(uint32_t);
    const size_t moduli_bytes = limb_count * sizeof(uint32_t);
    uint32_t* A_input_dev = nullptr;
    uint32_t* A_custom_dev = nullptr;
    uint32_t* twiddle_dev = nullptr;
    uint32_t* moduli_dev = nullptr;

    cudaMalloc((void**)&A_input_dev, bytes);
    cudaMalloc((void**)&A_custom_dev, bytes);
    cudaMalloc((void**)&twiddle_dev, bytes);
    cudaMalloc((void**)&moduli_dev, moduli_bytes);

    cudaMemcpy(A_input_dev, A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(twiddle_dev, twiddle_table.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(moduli_dev, moduli_host.data(), moduli_bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIM, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, N, logN,
                   limb_count, dimBlock);
    }
    cudaDeviceSynchronize();

    //copy time included
    profiler_range_push("custom_ntt");
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, N, logN,
                   limb_count, dimBlock);
    }
    cudaEventRecord(stop);

    cudaError_t launchErr = cudaGetLastError();
    cudaError_t syncErr = cudaEventSynchronize(stop);
    profiler_range_pop();

    float execution_time = 0.0f;
    if (launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cudaEventElapsedTime(&execution_time, start, stop);
        execution_time /= BENCHMARK_ITERATIONS;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (launchErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(launchErr) << std::endl;
    } else if (syncErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(syncErr) << std::endl;
    } else {
        print_time_result("Custom kernel", execution_time, ver_config);
    }

    if (verify && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cudaMemcpy(out_host, A_custom_dev, bytes, cudaMemcpyDeviceToHost);
        for (size_t limb = 0; limb < limb_count; ++limb) {
            cpu_ntt_butterfly(A_ref_host + limb * N, logN,
                              mod_runtime[limb].qi,
                              twiddle_table.data() + limb * N);
        }

        if (validate(A_ref_host, out_host, total_values)) {
            std::cout << ">>> Custom kernel test pass!" << std::endl;
        } else {
            std::cout << ">>> Custom kernel test fail!" << std::endl;
            std::cout << ">>> First 10 elements of answer:\n";
            print_first_10(A_ref_host, total_values);
            std::cout << ">>> First 10 elements of custom:\n";
            print_first_10(out_host, total_values);
        }

    } else if (verify) {
        std::cout << ">>> Verification skipped because the kernel did not complete successfully."
                  << std::endl;
    } else {
        std::cout << ">>> Verification skipped. Use --verify to enable it."
                  << std::endl;
    }

    cudaFree(A_input_dev);
    cudaFree(A_custom_dev);
    cudaFree(twiddle_dev);
    cudaFree(moduli_dev);

    delete[] A_host;
    delete[] A_ref_host;
    delete[] out_host;

    bool customOk = (launchErr == cudaSuccess && syncErr == cudaSuccess);
    bool allOk = customOk && !compare_base;
    return allOk ? 0 : 1;
}
