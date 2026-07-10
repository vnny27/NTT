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
    uint32_t psi_inv;
    uint32_t omega; // psi^2, primitive N-th root
    uint32_t degree_inv;
};

#define SEED 1234
#define WARMUP_ITERATIONS 10
#ifdef USE_NVTX
#define BENCHMARK_ITERATIONS 1
#else
#define BENCHMARK_ITERATIONS 30
#endif

#define BLOCK_DIM 256
#define MAX_PHASE_MERGE 32

__device__ inline void vectorized_load_u32_16(uint32_t* dst,
                                              const uint32_t* src) {
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4 v0 = src4[0];
    uint4 v1 = src4[1];
    uint4 v2 = src4[2];
    uint4 v3 = src4[3];
    dst[0] = v0.x;   dst[1] = v0.y;   dst[2] = v0.z;   dst[3] = v0.w;
    dst[4] = v1.x;   dst[5] = v1.y;   dst[6] = v1.z;   dst[7] = v1.w;
    dst[8] = v2.x;   dst[9] = v2.y;   dst[10] = v2.z;  dst[11] = v2.w;
    dst[12] = v3.x;  dst[13] = v3.y;  dst[14] = v3.z;  dst[15] = v3.w;
}

__device__ inline void vectorized_load_u32_8(uint32_t* dst,
                                             const uint32_t* src) {
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4 v0 = src4[0];
    uint4 v1 = src4[1];
    dst[0] = v0.x;  dst[1] = v0.y;  dst[2] = v0.z;  dst[3] = v0.w;
    dst[4] = v1.x;  dst[5] = v1.y;  dst[6] = v1.z;  dst[7] = v1.w;
}

__device__ inline void vectorized_load_u32_4(uint32_t* dst,
                                             const uint32_t* src) {
    uint4 v0 = reinterpret_cast<const uint4*>(src)[0];
    dst[0] = v0.x;  dst[1] = v0.y;  dst[2] = v0.z;  dst[3] = v0.w;
}

__device__ inline void vectorized_store_u32_16(uint32_t* dst,
                                               const uint32_t* src) {
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    dst4[0] = make_uint4(src[0], src[1], src[2], src[3]);
    dst4[1] = make_uint4(src[4], src[5], src[6], src[7]);
    dst4[2] = make_uint4(src[8], src[9], src[10], src[11]);
    dst4[3] = make_uint4(src[12], src[13], src[14], src[15]);
}

__device__ inline void vectorized_store_u32_8(uint32_t* dst,
                                              const uint32_t* src) {
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    dst4[0] = make_uint4(src[0], src[1], src[2], src[3]);
    dst4[1] = make_uint4(src[4], src[5], src[6], src[7]);
}

__device__ inline void vectorized_store_u32_4(uint32_t* dst,
                                              const uint32_t* src) {
    reinterpret_cast<uint4*>(dst)[0] =
        make_uint4(src[0], src[1], src[2], src[3]);
}

__device__ inline void vectorized_load_u32(uint32_t* dst,
                                           const uint32_t* src,
                                           size_t elems) {
    if (elems == 16) {
        vectorized_load_u32_16(dst, src);
    } else if (elems == 8) {
        vectorized_load_u32_8(dst, src);
    } else if (elems == 4) {
        vectorized_load_u32_4(dst, src);
    } else {
        for (size_t i = 0; i < elems; ++i) {
            dst[i] = src[i];
        }
    }
}

__device__ inline void vectorized_store_u32(uint32_t* dst,
                                            const uint32_t* src,
                                            size_t elems) {
    if (elems == 16) {
        vectorized_store_u32_16(dst, src);
    } else if (elems == 8) {
        vectorized_store_u32_8(dst, src);
    } else if (elems == 4) {
        vectorized_store_u32_4(dst, src);
    } else {
        for (size_t i = 0; i < elems; ++i) {
            dst[i] = src[i];
        }
    }
}

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

__device__ inline void inverse_butterfly(uint32_t* A_limb,
                                         const uint32_t* tw_limb,
                                         size_t t, size_t stage_block,
                                         size_t stage_id, size_t m,
                                         uint32_t q) {
    size_t a = stage_block * 2 * t + stage_id;
    uint32_t u = A_limb[a];
    uint32_t v = A_limb[a + t];
    A_limb[a] = add_mod(u, v, q);
    A_limb[a + t] = mul_mod(sub_mod(u, v, q), tw_limb[m + stage_block], q);
}

__device__ inline void butterfly_merge_stages(uint32_t* local,
                                              const uint32_t* tw_limb,
                                              uint32_t q, size_t merge,
                                              size_t stages,
                                              size_t tw_base) {
    for (size_t stage = 0, m = 1, t = merge >> 1; stage < stages;
         stage++, m <<= 1, t >>= 1) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = j * 2 * t; k < j * 2 * t + t; ++k) {
                uint32_t tw = tw_limb[(tw_base << stage) + j];
                uint32_t u = local[k];
                uint32_t v = mul_mod(local[k + t], tw, q);
                local[k] = add_mod(u, v, q);
                local[k + t] = sub_mod(u, v, q);
            }
        }
    }
}

__device__ inline void inverse_butterfly_merge_stages(uint32_t* local,
                                                      const uint32_t* tw_limb,
                                                      uint32_t q,
                                                      size_t merge,
                                                      size_t stages,
                                                      size_t tw_base) {
    for (size_t stage = 0, m = size_t{1} << (stages - 1), t = merge >> stages;
         stage < stages; stage++, m >>= 1, t <<= 1) {
        size_t tw_shift = stages - 1 - stage;
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = j * 2 * t; k < j * 2 * t + t; ++k) {
                uint32_t tw = tw_limb[(tw_base << tw_shift) + j];
                uint32_t u = local[k];
                uint32_t v = local[k + t];
                local[k] = add_mod(u, v, q);
                local[k + t] = mul_mod(sub_mod(u, v, q), tw, q);
            }
        }
    }
}

__global__ void NTTPhase1(uint32_t* A, const uint32_t* twiddles,
                          const uint32_t* moduli, int logN,
                          int radix_stages, int stage_merging,
                          int warp_batching) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t N = uint64_t{1} << logN;
    size_t merge = size_t{1} << stage_merging;
    size_t batch = size_t{1} << warp_batching;
    size_t merge_log_stride = logN - stage_merging;
    size_t radix_log_stride = logN - radix_stages;
    extern __shared__ uint32_t A_radix[];
    uint32_t local[MAX_PHASE_MERGE];
    if (merge > MAX_PHASE_MERGE) {
        return;
    }

    int limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = twiddles + limb * N;

    size_t batch_block = static_cast<size_t>(tid) >> warp_batching;
    size_t batch_id = static_cast<size_t>(tid) & (batch - 1);

    size_t tail_stages = (radix_stages - 1) % stage_merging + 1;
    size_t merge_stages = (radix_stages - 1) / stage_merging;

    size_t input_base =
        (batch_block << radix_log_stride) +
        (static_cast<size_t>(bid) << warp_batching) +
        batch_id;

    for (size_t i = 0; i < merge; i++) {
        local[i] = A_limb[input_base + (i << merge_log_stride)];
    }

    butterfly_merge_stages(local, tw_limb, q, merge, tail_stages, 1);

    size_t tid_s = static_cast<size_t>(tid);
    if (merge_stages > 0) {
        size_t sm_log_stride = radix_stages - stage_merging + warp_batching;
        for (size_t i = 0; i < merge; i++) {
            A_radix[tid_s + (i << sm_log_stride)] = local[i];
        }

        __syncthreads();

        sm_log_stride -= tail_stages;
        for (size_t i = 0; i + 1 < merge_stages; i++) {
            size_t sm_base =
                ((tid_s >> sm_log_stride) << (sm_log_stride + stage_merging)) +
                (tid_s & ((size_t{1} << sm_log_stride) - 1));
            for (size_t j = 0; j < merge; j++) {
                local[j] = A_radix[sm_base + (j << sm_log_stride)];
            }

            size_t tw_base = (size_t{1} << (tail_stages + i * stage_merging))
                             + (tid_s >> sm_log_stride);
            butterfly_merge_stages(local, tw_limb, q, merge, stage_merging,
                                   tw_base);

            for (size_t j = 0; j < merge; j++) {
                A_radix[sm_base + (j << sm_log_stride)] = local[j];
            }
            sm_log_stride -= stage_merging;
            __syncthreads();
        }

        size_t i = merge_stages - 1;
        size_t sm_base =
            ((tid_s >> sm_log_stride) << (sm_log_stride + stage_merging)) +
            (tid_s & ((size_t{1} << sm_log_stride) - 1));
        for (size_t j = 0; j < merge; j++) {
            local[j] = A_radix[sm_base + (j << sm_log_stride)];
        }

        size_t tw_base = (size_t{1} << (tail_stages + i * stage_merging))
                         + (tid_s >> sm_log_stride);
        butterfly_merge_stages(local, tw_limb, q, merge, stage_merging,
                               tw_base);
    }

    size_t dst_base =
        batch_id +
        (batch_block << ((logN - radix_stages) + stage_merging)) +
        (static_cast<size_t>(bid) << warp_batching);
    for (size_t i = 0; i < merge; i++) {
        A_limb[dst_base + (i << radix_log_stride)] = local[i];
    }
}

__global__ void NTTPhase2(uint32_t* A, const uint32_t* twiddles,
                          const uint32_t* moduli, int logN,
                          int radix_stages, int stage_merging,
                          int warp_batching) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t N = uint64_t{1} << logN;
    size_t merge = size_t{1} << stage_merging;
    size_t ph1_radix = size_t{1} << (logN - radix_stages);
    size_t group_log_width = radix_stages - stage_merging;
    size_t group_width = size_t{1} << group_log_width;
    size_t groups_in_block = static_cast<size_t>(blockDim.x) >> group_log_width;
    extern __shared__ uint32_t A_radix[];
    uint32_t local[MAX_PHASE_MERGE];
    if (merge > MAX_PHASE_MERGE) {
        return;
    }
    (void)warp_batching;

    int limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = twiddles + limb * N;

    size_t tid_s = static_cast<size_t>(tid);
    size_t group_id = tid_s >> group_log_width;
    size_t group_element = tid_s & (group_width - 1);
    size_t global_segment = static_cast<size_t>(bid) * groups_in_block + group_id;
    size_t segment_offset = global_segment << radix_stages;

    size_t tail_stages = (radix_stages - 1) % stage_merging + 1;
    size_t merge_stages = (radix_stages - 1) / stage_merging;

    size_t input_base = segment_offset + group_element;

    for (size_t i = 0; i < merge; i++) {
        local[i] = A_limb[input_base + (i << group_log_width)];
    }

    butterfly_merge_stages(local, tw_limb, q, merge, tail_stages,
                           ph1_radix + global_segment);

    size_t sm_segment_offset = group_id << radix_stages;
    if (merge_stages > 0) {
        size_t sm_log_stride = radix_stages - stage_merging;
        for (size_t i = 0; i < merge; i++) {
            A_radix[sm_segment_offset + group_element +
                    (i << sm_log_stride)] = local[i];
        }

        __syncthreads();

        sm_log_stride -= tail_stages;
        for (size_t i = 0; i + 1 < merge_stages; i++) {
            size_t sm_base =
                sm_segment_offset +
                ((group_element >> sm_log_stride)
                 << (sm_log_stride + stage_merging)) +
                (group_element & ((size_t{1} << sm_log_stride) - 1));
            for (size_t j = 0; j < merge; j++) {
                local[j] = A_radix[sm_base + (j << sm_log_stride)];
            }

            size_t tw_base =
                ((ph1_radix + global_segment)
                 << (tail_stages + i * stage_merging)) +
                (group_element >> sm_log_stride);
            butterfly_merge_stages(local, tw_limb, q, merge, stage_merging,
                                   tw_base);

            for (size_t j = 0; j < merge; j++) {
                A_radix[sm_base + (j << sm_log_stride)] = local[j];
            }
            sm_log_stride -= stage_merging;
            __syncthreads();
        }

        size_t i = merge_stages - 1;
        size_t sm_base =
            sm_segment_offset +
            ((group_element >> sm_log_stride)
             << (sm_log_stride + stage_merging)) +
            (group_element & ((size_t{1} << sm_log_stride) - 1));
        for (size_t j = 0; j < merge; j++) {
            local[j] = A_radix[sm_base + (j << sm_log_stride)];
        }
        __syncthreads();

        size_t tw_base =
            ((ph1_radix + global_segment)
             << (tail_stages + i * stage_merging)) +
            (group_element >> sm_log_stride);
        butterfly_merge_stages(local, tw_limb, q, merge, stage_merging,
                               tw_base);
    }

    uint32_t* dst_ptr =
        A_limb + segment_offset + (group_element << stage_merging);
    vectorized_store_u32(dst_ptr, local, merge);
}

__global__ void INTTPhase1(uint32_t* A, const uint32_t* inv_twiddles,
                           const uint32_t* moduli, int logN,
                           int radix_stages, int stage_merging,
                           int warp_batching) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t N = uint64_t{1} << logN;
    size_t merge = size_t{1} << stage_merging;
    size_t ph2_radix = size_t{1} << (logN - radix_stages);
    size_t group_log_width = radix_stages - stage_merging;
    size_t group_width = size_t{1} << group_log_width;
    size_t groups_in_block = static_cast<size_t>(blockDim.x) >> group_log_width;
    extern __shared__ uint32_t A_radix[];
    uint32_t local[MAX_PHASE_MERGE];
    if (merge > MAX_PHASE_MERGE) {
        return;
    }
    (void)warp_batching;

    int limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = inv_twiddles + limb * N;

    size_t tid_s = static_cast<size_t>(tid);
    size_t group_id = tid_s >> group_log_width;
    size_t group_element = tid_s & (group_width - 1);
    size_t global_segment = static_cast<size_t>(bid) * groups_in_block + group_id;
    size_t segment_offset = global_segment << radix_stages;

    size_t tail_stages = (radix_stages - 1) % stage_merging + 1;
    size_t merge_stages = (radix_stages - 1) / stage_merging;

    const uint32_t* src_ptr =
        A_limb + segment_offset + (group_element << stage_merging);
    vectorized_load_u32(local, src_ptr, merge);

    size_t sm_segment_offset = group_id << radix_stages;
    if (merge_stages > 0) {
        size_t sm_log_stride = 0;
        for (size_t i = merge_stages; i > 0; i--) {
            size_t merge_stage = i - 1;
            size_t sm_base =
                sm_segment_offset +
                ((group_element >> sm_log_stride)
                 << (sm_log_stride + stage_merging)) +
                (group_element & ((size_t{1} << sm_log_stride) - 1));
            size_t tw_base =
                ((ph2_radix + global_segment)
                 << (tail_stages + merge_stage * stage_merging)) +
                (group_element >> sm_log_stride);

            inverse_butterfly_merge_stages(local, tw_limb, q, merge,
                                           stage_merging, tw_base);

            for (size_t j = 0; j < merge; j++) {
                A_radix[sm_base + (j << sm_log_stride)] = local[j];
            }
            __syncthreads();

            if (merge_stage > 0) {
                sm_log_stride += stage_merging;
                sm_base =
                    sm_segment_offset +
                    ((group_element >> sm_log_stride)
                     << (sm_log_stride + stage_merging)) +
                    (group_element & ((size_t{1} << sm_log_stride) - 1));
                for (size_t j = 0; j < merge; j++) {
                    local[j] = A_radix[sm_base + (j << sm_log_stride)];
                }
            }
        }

        size_t tail_sm_log_stride = sm_log_stride + tail_stages;
        size_t sm_base =
            sm_segment_offset +
            ((group_element >> tail_sm_log_stride)
             << (tail_sm_log_stride + stage_merging)) +
            (group_element & ((size_t{1} << tail_sm_log_stride) - 1));
        for (size_t j = 0; j < merge; j++) {
            local[j] = A_radix[sm_base + (j << tail_sm_log_stride)];
        }
    }

    inverse_butterfly_merge_stages(local, tw_limb, q, merge, tail_stages,
                                   ph2_radix + global_segment);

    size_t output_base = segment_offset + group_element;
    for (size_t i = 0; i < merge; i++) {
        A_limb[output_base + (i << group_log_width)] = local[i];
    }
}

__global__ void INTTPhase2(uint32_t* A, const uint32_t* inv_twiddles,
                           const uint32_t* moduli, int logN,
                           int radix_stages, int stage_merging,
                           int warp_batching) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint64_t N = uint64_t{1} << logN;
    size_t merge = size_t{1} << stage_merging;
    size_t batch = size_t{1} << warp_batching;
    size_t merge_log_stride = logN - stage_merging;
    size_t radix_log_stride = logN - radix_stages;
    extern __shared__ uint32_t A_radix[];
    uint32_t local[MAX_PHASE_MERGE];
    if (merge > MAX_PHASE_MERGE) {
        return;
    }

    int limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = inv_twiddles + limb * N;

    size_t batch_block = static_cast<size_t>(tid) >> warp_batching;
    size_t batch_id = static_cast<size_t>(tid) & (batch - 1);

    size_t tail_stages = (radix_stages - 1) % stage_merging + 1;
    size_t merge_stages = (radix_stages - 1) / stage_merging;

    size_t input_base =
        (batch_block << radix_log_stride) +
        (static_cast<size_t>(bid) << warp_batching) +
        batch_id;
    size_t output_base =
        batch_id +
        (batch_block << ((logN - radix_stages) + stage_merging)) +
        (static_cast<size_t>(bid) << warp_batching);

    for (size_t i = 0; i < merge; i++) {
        local[i] = A_limb[output_base + (i << radix_log_stride)];
    }

    size_t tid_s = static_cast<size_t>(tid);
    if (merge_stages > 0) {
        size_t sm_log_stride = warp_batching;
        for (size_t i = merge_stages; i > 0; i--) {
            size_t merge_stage = i - 1;
            size_t sm_base =
                ((tid_s >> sm_log_stride) << (sm_log_stride + stage_merging)) +
                (tid_s & ((size_t{1} << sm_log_stride) - 1));
            size_t tw_base =
                (size_t{1} << (tail_stages + merge_stage * stage_merging)) +
                (tid_s >> sm_log_stride);

            inverse_butterfly_merge_stages(local, tw_limb, q, merge,
                                           stage_merging, tw_base);

            for (size_t j = 0; j < merge; j++) {
                A_radix[sm_base + (j << sm_log_stride)] = local[j];
            }
            __syncthreads();

            if (merge_stage > 0) {
                sm_log_stride += stage_merging;
                sm_base =
                    ((tid_s >> sm_log_stride)
                     << (sm_log_stride + stage_merging)) +
                    (tid_s & ((size_t{1} << sm_log_stride) - 1));
                for (size_t j = 0; j < merge; j++) {
                    local[j] = A_radix[sm_base + (j << sm_log_stride)];
                }
            }
        }

        size_t tail_sm_log_stride = sm_log_stride + tail_stages;
        size_t sm_base =
            ((tid_s >> tail_sm_log_stride)
             << (tail_sm_log_stride + stage_merging)) +
            (tid_s & ((size_t{1} << tail_sm_log_stride) - 1));
        for (size_t j = 0; j < merge; j++) {
            local[j] = A_radix[sm_base + (j << tail_sm_log_stride)];
        }
    }

    inverse_butterfly_merge_stages(local, tw_limb, q, merge, tail_stages, 1);

    for (size_t i = 0; i < merge; i++) {
        A_limb[input_base + (i << merge_log_stride)] = local[i];
    }
}

__global__ void INTTStage(uint32_t* A, const uint32_t* inv_twiddles,
                          const uint32_t* moduli, size_t N,
                          size_t m, size_t t) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (N >> 1)) {
        return;
    }

    size_t limb = blockIdx.y;
    uint32_t q = moduli[limb];
    uint32_t* A_limb = A + limb * N;
    const uint32_t* tw_limb = inv_twiddles + limb * N;

    size_t stage_block = tid / t;
    size_t stage_id = tid % t;
    inverse_butterfly(A_limb, tw_limb, t, stage_block, stage_id, m, q);
}

__global__ void NormalizeKernel(uint32_t* A, const uint32_t* moduli,
                                const uint32_t* degree_inv, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) {
        return;
    }

    size_t limb = blockIdx.y;
    uint32_t q = moduli[limb];
    A[limb * N + tid] = mul_mod(A[limb * N + tid], degree_inv[limb], q);
}

dim3 phase_block_dim(const PhaseConfig& phase) {
    int block_log_width =
        phase.radix_stages - phase.stage_merging + phase.warp_batching;
    return dim3(1u << block_log_width);
}

dim3 phase_grid_dim(size_t N, size_t limb_count, const PhaseConfig& phase,
                    dim3 block) {
    size_t phase_threads = N >> phase.stage_merging;
    unsigned int grid_x =
        static_cast<unsigned int>((phase_threads + block.x - 1) / block.x);
    return dim3(grid_x, static_cast<unsigned int>(limb_count), 1);
}

size_t phase_shared_bytes(const PhaseConfig& phase, dim3 block) {
    return (static_cast<size_t>(block.x) << phase.stage_merging) *
           sizeof(uint32_t);
}

void launch_ntt(uint32_t* A_dev, const uint32_t* twiddle_dev,
                const uint32_t* moduli_dev, const VersionConfig& config,
                size_t limb_count) {
    size_t N = size_t{1} << config.logN;
    const PhaseConfig& phase1 = config.transform.ntt_phase1;
    const PhaseConfig& phase2 = config.transform.ntt_phase2;

    dim3 phase1_block = phase_block_dim(phase1);
    dim3 phase1_grid = phase_grid_dim(N, limb_count, phase1, phase1_block);
    size_t phase1_shared = phase_shared_bytes(phase1, phase1_block);
    NTTPhase1<<<phase1_grid, phase1_block, phase1_shared>>>(
        A_dev, twiddle_dev, moduli_dev, config.logN, phase1.radix_stages,
        phase1.stage_merging, phase1.warp_batching);

    dim3 phase2_block = phase_block_dim(phase2);
    dim3 phase2_grid = phase_grid_dim(N, limb_count, phase2, phase2_block);
    size_t phase2_shared = phase_shared_bytes(phase2, phase2_block);
    NTTPhase2<<<phase2_grid, phase2_block, phase2_shared>>>(
        A_dev, twiddle_dev, moduli_dev, config.logN, phase2.radix_stages,
        phase2.stage_merging, phase2.warp_batching);
}

void launch_intt(uint32_t* A_dev, const uint32_t* inv_twiddle_dev,
                 const uint32_t* moduli_dev, const uint32_t* degree_inv_dev,
                 const VersionConfig& config, size_t limb_count) {
    size_t N = size_t{1} << config.logN;
    const PhaseConfig& phase1 = config.transform.intt_phase1;
    const PhaseConfig& phase2 = config.transform.intt_phase2;

    dim3 phase1_block = phase_block_dim(phase1);
    dim3 phase1_grid = phase_grid_dim(N, limb_count, phase1, phase1_block);
    size_t phase1_shared = phase_shared_bytes(phase1, phase1_block);
    INTTPhase1<<<phase1_grid, phase1_block, phase1_shared>>>(
        A_dev, inv_twiddle_dev, moduli_dev, config.logN, phase1.radix_stages,
        phase1.stage_merging, phase1.warp_batching);

    dim3 phase2_block = phase_block_dim(phase2);
    dim3 phase2_grid = phase_grid_dim(N, limb_count, phase2, phase2_block);
    size_t phase2_shared = phase_shared_bytes(phase2, phase2_block);
    INTTPhase2<<<phase2_grid, phase2_block, phase2_shared>>>(
        A_dev, inv_twiddle_dev, moduli_dev, config.logN, phase2.radix_stages,
        phase2.stage_merging, phase2.warp_batching);

    dim3 normalizeBlock(BLOCK_DIM);
    unsigned int normalize_grid_x =
        static_cast<unsigned int>((N + normalizeBlock.x - 1) / normalizeBlock.x);
    dim3 normalizeGrid(normalize_grid_x, static_cast<unsigned int>(limb_count), 1);
    NormalizeKernel<<<normalizeGrid, normalizeBlock>>>(
        A_dev, moduli_dev, degree_inv_dev, N);
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
    uint32_t psi_inv = pow_mod(psi, config.qi - 2, config.qi);
    uint32_t omega = static_cast<uint32_t>(
        (static_cast<uint64_t>(psi) * psi) % config.qi);
    uint32_t degree_inv = pow_mod(static_cast<uint32_t>(N), config.qi - 2,
                                  config.qi);

    if (pow_mod(psi, root_order, config.qi) != 1 ||
        pow_mod(psi, N, config.qi) != config.qi - 1) {
        std::cerr << "primitive_root did not generate a primitive 2N-th root.\n";
        std::exit(EXIT_FAILURE);
    }

    return {config.qi, config.primitive_root, psi, psi_inv, omega, degree_inv};
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

void cpu_intt_butterfly(uint32_t* a, int logN, uint32_t q,
                        const uint32_t* inv_psi, uint32_t degree_inv) {
    size_t N = size_t{1} << logN;
    size_t t = 1;

    for (size_t m = N >> 1; m >= 1; m >>= 1) {
        for (size_t j = 0; j < m; j += 1) {
            for (size_t k = j * 2 * t; k < j * 2 * t + t; k += 1) {
                uint32_t u = a[k];
                uint32_t v = a[k + t];
                a[k] = add_mod(u, v, q);
                a[k + t] = mul_mod(sub_mod(u, v, q), inv_psi[m + j], q);
            }
        }
        t <<= 1;
        if (m == 1) {
            break;
        }
    }

    for (size_t i = 0; i < N; ++i) {
        a[i] = mul_mod(a[i], degree_inv, q);
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
    std::vector<uint32_t> degree_inv_host(limb_count);
    std::vector<uint32_t> twiddle_table(limb_count * N);
    std::vector<uint32_t> inv_twiddle_table(limb_count * N);
    mod_runtime.reserve(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb) {
        ModulusRuntime runtime =
            make_modulus_runtime(logN, ver_config.moduli[limb]);
        mod_runtime.push_back(runtime);
        moduli_host[limb] = runtime.qi;
        degree_inv_host[limb] = runtime.degree_inv;

        std::vector<uint32_t> limb_twiddles =
            make_twiddle_table(logN, runtime.qi, runtime.psi);
        std::vector<uint32_t> limb_inv_twiddles =
            make_twiddle_table(logN, runtime.qi, runtime.psi_inv);
        std::copy(limb_twiddles.begin(), limb_twiddles.end(),
                  twiddle_table.begin() + limb * N);
        std::copy(limb_inv_twiddles.begin(), limb_inv_twiddles.end(),
                  inv_twiddle_table.begin() + limb * N);
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
    uint32_t* A_ntt_input_dev = nullptr;
    uint32_t* twiddle_dev = nullptr;
    uint32_t* inv_twiddle_dev = nullptr;
    uint32_t* moduli_dev = nullptr;
    uint32_t* degree_inv_dev = nullptr;

    cudaMalloc((void**)&A_input_dev, bytes);
    cudaMalloc((void**)&A_custom_dev, bytes);
    cudaMalloc((void**)&A_ntt_input_dev, bytes);
    cudaMalloc((void**)&twiddle_dev, bytes);
    cudaMalloc((void**)&inv_twiddle_dev, bytes);
    cudaMalloc((void**)&moduli_dev, moduli_bytes);
    cudaMalloc((void**)&degree_inv_dev, moduli_bytes);

    cudaMemcpy(A_input_dev, A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(twiddle_dev, twiddle_table.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(inv_twiddle_dev, inv_twiddle_table.data(), bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(moduli_dev, moduli_host.data(), moduli_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(degree_inv_dev, degree_inv_host.data(), moduli_bytes,
               cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, ver_config,
                   limb_count);
    }
    cudaDeviceSynchronize();

    //copy time included
    profiler_range_push("custom_ntt");
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, ver_config,
                   limb_count);
    }
    cudaEventRecord(stop);

    cudaError_t nttLaunchErr = cudaGetLastError();
    cudaError_t nttSyncErr = cudaEventSynchronize(stop);
    profiler_range_pop();

    float ntt_execution_time = 0.0f;
    if (nttLaunchErr == cudaSuccess && nttSyncErr == cudaSuccess) {
        cudaEventElapsedTime(&ntt_execution_time, start, stop);
        ntt_execution_time /= BENCHMARK_ITERATIONS;
    }

    if (nttLaunchErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(nttLaunchErr)
                  << std::endl;
    } else if (nttSyncErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(nttSyncErr)
                  << std::endl;
    } else {
        print_time_result("Custom NTT kernel", ntt_execution_time, ver_config);
    }

    cudaMemcpy(A_ntt_input_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
    launch_ntt(A_ntt_input_dev, twiddle_dev, moduli_dev, ver_config,
               limb_count);
    cudaError_t prepLaunchErr = cudaGetLastError();
    cudaError_t prepSyncErr = cudaDeviceSynchronize();

    cudaError_t inttLaunchErr = prepLaunchErr;
    cudaError_t inttSyncErr = prepSyncErr;
    float intt_execution_time = 0.0f;
    if (prepLaunchErr == cudaSuccess && prepSyncErr == cudaSuccess) {
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            cudaMemcpy(A_custom_dev, A_ntt_input_dev, bytes,
                       cudaMemcpyDeviceToDevice);
            launch_intt(A_custom_dev, inv_twiddle_dev, moduli_dev,
                        degree_inv_dev, ver_config, limb_count);
        }
        cudaDeviceSynchronize();

        //copy time included
        profiler_range_push("custom_intt");
        cudaEventRecord(start);
        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            cudaMemcpy(A_custom_dev, A_ntt_input_dev, bytes,
                       cudaMemcpyDeviceToDevice);
            launch_intt(A_custom_dev, inv_twiddle_dev, moduli_dev,
                        degree_inv_dev, ver_config, limb_count);
        }
        cudaEventRecord(stop);

        inttLaunchErr = cudaGetLastError();
        inttSyncErr = cudaEventSynchronize(stop);
        profiler_range_pop();

        if (inttLaunchErr == cudaSuccess && inttSyncErr == cudaSuccess) {
            cudaEventElapsedTime(&intt_execution_time, start, stop);
            intt_execution_time /= BENCHMARK_ITERATIONS;
        }
    }

    if (inttLaunchErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(inttLaunchErr)
                  << std::endl;
    } else if (inttSyncErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(inttSyncErr)
                  << std::endl;
    } else {
        print_time_result("Custom INTT kernel", intt_execution_time, ver_config);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (verify && nttLaunchErr == cudaSuccess && nttSyncErr == cudaSuccess &&
        inttLaunchErr == cudaSuccess && inttSyncErr == cudaSuccess) {
        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, ver_config,
                   limb_count);
        cudaError_t verifyNttLaunchErr = cudaGetLastError();
        cudaError_t verifyNttSyncErr = cudaDeviceSynchronize();

        std::copy(A_host, A_host + total_values, A_ref_host);
        for (size_t limb = 0; limb < limb_count; ++limb) {
            cpu_ntt_butterfly(A_ref_host + limb * N, logN,
                              mod_runtime[limb].qi,
                              twiddle_table.data() + limb * N);
        }

        bool nttOk = false;
        if (verifyNttLaunchErr == cudaSuccess && verifyNttSyncErr == cudaSuccess) {
            cudaMemcpy(out_host, A_custom_dev, bytes, cudaMemcpyDeviceToHost);
            nttOk = validate(A_ref_host, out_host, total_values);
        }

        if (nttOk) {
            std::cout << ">>> Custom NTT test pass!" << std::endl;
        } else {
            std::cout << ">>> Custom NTT test fail!" << std::endl;
            if (verifyNttLaunchErr != cudaSuccess) {
                std::cout << "  [CUDA ERROR]: "
                          << cudaGetErrorString(verifyNttLaunchErr)
                          << std::endl;
            } else if (verifyNttSyncErr != cudaSuccess) {
                std::cout << "  [CUDA ERROR]: "
                          << cudaGetErrorString(verifyNttSyncErr)
                          << std::endl;
            }
            std::cout << ">>> First 10 elements of answer:\n";
            print_first_10(A_ref_host, total_values);
            std::cout << ">>> First 10 elements of custom:\n";
            print_first_10(out_host, total_values);
        }

        cudaMemcpy(A_custom_dev, A_input_dev, bytes, cudaMemcpyDeviceToDevice);
        launch_ntt(A_custom_dev, twiddle_dev, moduli_dev, ver_config,
                   limb_count);
        launch_intt(A_custom_dev, inv_twiddle_dev, moduli_dev, degree_inv_dev,
                    ver_config, limb_count);
        cudaError_t verifyInttLaunchErr = cudaGetLastError();
        cudaError_t verifyInttSyncErr = cudaDeviceSynchronize();

        std::copy(A_host, A_host + total_values, A_ref_host);
        for (size_t limb = 0; limb < limb_count; ++limb) {
            cpu_ntt_butterfly(A_ref_host + limb * N, logN,
                              mod_runtime[limb].qi,
                              twiddle_table.data() + limb * N);
            cpu_intt_butterfly(A_ref_host + limb * N, logN,
                               mod_runtime[limb].qi,
                               inv_twiddle_table.data() + limb * N,
                               mod_runtime[limb].degree_inv);
        }

        bool inttOk = false;
        if (verifyInttLaunchErr == cudaSuccess &&
            verifyInttSyncErr == cudaSuccess) {
            cudaMemcpy(out_host, A_custom_dev, bytes, cudaMemcpyDeviceToHost);
            inttOk = validate(A_ref_host, out_host, total_values);
        }

        if (inttOk) {
            std::cout << ">>> Custom INTT roundtrip test pass!" << std::endl;
        } else {
            std::cout << ">>> Custom INTT roundtrip test fail!" << std::endl;
            if (verifyInttLaunchErr != cudaSuccess) {
                std::cout << "  [CUDA ERROR]: "
                          << cudaGetErrorString(verifyInttLaunchErr)
                          << std::endl;
            } else if (verifyInttSyncErr != cudaSuccess) {
                std::cout << "  [CUDA ERROR]: "
                          << cudaGetErrorString(verifyInttSyncErr)
                          << std::endl;
            }
            std::cout << ">>> First 10 elements of answer:\n";
            print_first_10(A_ref_host, total_values);
            std::cout << ">>> First 10 elements of custom:\n";
            print_first_10(out_host, total_values);
        }

    } else if (verify) {
        std::cout << ">>> Verification skipped because the kernel did not complete "
                     "successfully."
                  << std::endl;
    } else {
        std::cout << ">>> Verification skipped. Use --verify to enable it."
                  << std::endl;
    }

    cudaFree(A_input_dev);
    cudaFree(A_custom_dev);
    cudaFree(A_ntt_input_dev);
    cudaFree(twiddle_dev);
    cudaFree(inv_twiddle_dev);
    cudaFree(moduli_dev);
    cudaFree(degree_inv_dev);

    delete[] A_host;
    delete[] A_ref_host;
    delete[] out_host;

    bool customOk = (nttLaunchErr == cudaSuccess && nttSyncErr == cudaSuccess &&
                     inttLaunchErr == cudaSuccess && inttSyncErr == cudaSuccess);
    bool allOk = customOk && !compare_base;
    return allOk ? 0 : 1;
}
