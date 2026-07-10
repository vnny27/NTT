# CUDA NTT Benchmark Notes

This directory tracks CUDA NTT baseline and optimization versions.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090  |
| Driver version | 590.48.01 |
| CUDA version | 13.1 |

## Versions

`src/v00_base_ntt.cu` is the first correctness-oriented GPU baseline.
`src/v01_radix_stages.cu` is the first phase-split radix-staged version.
`src/v02_vectorized_io.cu` adds vectorized phase-boundary IO to v01.
`src/v03_template_config.cu` specializes the phase config with templates.
`src/v04_montgomery.cu` uses Montgomery multiplication with fused normal input
conversion in NTT Phase1.

- Data type: `uint32_t`
- Transform size: `N = 1 << logN`
- Modulus layout: multi-limb RNS-style layout, `A[limb * N + i]`
- Verification: CPU butterfly NTT and NTT-to-INTT roundtrip

Kernel shape by version:

- `v00`: one CUDA kernel launch per radix-2 NTT/INTT stage
- `v01`: two phase kernels for NTT/INTT, with stage merging inside each phase
- `v02`: v01 plus vectorized NTT Phase2 store and INTT Phase1 load
- `v03`: v02 plus template-specialized launch config and phase kernels
- `v04`: v03 plus Montgomery-domain twiddles and modular multiplication

The current benchmark restores the device input before each measured transform.
The reported time includes that device-to-device restore copy.
For `v04`, NTT starts from normal-domain input and converts it to Montgomery form
inside NTT Phase1. INTT includes the final Montgomery-to-normal conversion.

## Build

From this directory:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 \
  src/<version>.cu -o <version>
```

For example, `src/v01_radix_stages.cu` builds to `./v01_radix_stages`.

`src/cheddar_ntt_bench.cu` is a separate timing-only comparison against
Cheddar's `NTTHandler`. The local CMake target builds only the Cheddar NTT
sources needed for timing, so GMP/libtommath is not required.

Cheddar comparison build:

```bash
cmake -S . -B build-cheddar -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cheddar --target cheddar_ntt_bench -j
./build-cheddar/cheddar_ntt_bench --verify
```

With NVTX ranges for Nsight:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -DUSE_NVTX \
  -I /usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3 \
  src/<version>.cu -o <version>
```

When `USE_NVTX` is enabled, `BENCHMARK_ITERATIONS` is set to `1` to keep
profiling traces compact.

Adjust `sm_89` if the target GPU uses a different architecture.

## Run

Benchmark only:

```bash
./<version>
```

Benchmark and verify against the CPU reference:

```bash
./<version> --verify
```

Short aliases:

```bash
./<version> -v
./<version> -h
```

## Config

Edit the shared config in `src/ntt_config.h`:

```cpp
using LaunchConfig = DefaultNTTLaunchConfig<16>;
VersionConfig ver_config = make_version_config<LaunchConfig>();
```

`DefaultNTTLaunchConfig<LogN>` derives the phase config from `LogN`:

```cpp
NTT Phase1 / INTT Phase2:
    radix_stages = LogN == 16 ? 7 : LogN - 9
    stage_merging = LogN == 16 ? 4 : 3
    warp_batching = 4

NTT Phase2 / INTT Phase1:
    radix_stages = 9
    stage_merging = 3
    warp_batching = 0
```

The modulus limbs live in `default_moduli()`:

```cpp
inline std::vector<ModulusConfig> default_moduli() {
    return {
        {974258177u, 3u},
        // ...
        {1016463361u, 38u},
    };
}
```

Each `qi` must satisfy:

```text
qi == 1 mod 2N
```

`primitive_root` is a primitive root modulo `qi`. The code derives:

- `psi = primitive_root^((qi - 1) / (2N)) mod qi`
- `omega = psi^2 mod qi`

To add more limbs, add more entries to `moduli`:

```cpp
{qi_next, primitive_root_next},
```

Inputs, outputs, twiddles, and CPU verification are then handled per limb.

For each phase config, the fields are:

```cpp
{radix_stages, stage_merging, warp_batching}
```

- `radix_stages`: `log2(radix)` handled by the phase.
- `stage_merging`: radix-2 stages fused inside one local register/shared-memory step.
- `warp_batching`: batching factor used by phase indexing to improve coalescing.

`v01` and `v02` consume these values at runtime through `VersionConfig`.
`v03` and `v04` use the same config as compile-time template parameters.

## Montgomery Variant

`src/v04_montgomery.cu` keeps twiddles and `degree_inv` in Montgomery form.
The input stored on the device is normal-domain input. NTT Phase1 converts each
loaded coefficient with:

```cpp
mont_mul(x, R^2 mod qi)
```

because Montgomery multiplication returns `a * b * R^-1 mod qi`, so
`mont_mul(x, R^2)` produces `x * R mod qi`.

The NTT result remains in Montgomery form. INTT applies `degree_inv` in
Montgomery form and then converts the final output back to normal domain.

## Profiling

Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx --stats=true \
  -o ntt_<version>_nsys ./<version> --verify
```

Nsight Compute:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all \
  --nvtx --nvtx-include "custom_ntt/" \
  -o ntt_<version>_ncu ./<version>
```

## Results Log

The benchmark time currently includes restoring the device input before each transform.
`default` refers to the config in `src/ntt_config.h`.

| Version | Config | Limbs | NTT Time (ms) | NTT GOp/s | INTT Time (ms) | INTT GOp/s | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v00 | default | 16 | 0.078 | 323.995 | 0.089 | 283.712 | Per-stage kernel |
| v01 | default | 16 | 0.067 | 378.092 | 0.073 | 346.141 | Two-phase radix-staged |
| v02 | default | 16 | 0.066 | 384.00 | 0.073 | 344.926 | Vectorized phase-boundary IO |
| v03 | default | 16 | 0.044 | 571.535 | 0.051 | 497.427 | Template-specialized phase config |
| v04 | default | 16 | 0.017 | 1461.769 | 0.020 | 12420.429 | Montgomery arithmetic with fused input conversion |

## Version Notes

Use this pattern when adding later versions:

- `v00_base_ntt.cu`: naive per-stage GPU baseline
- `v01_radix_stages.cu`: phase-split radix-staged kernel
- `v02_vectorized_io.cu`: v01 plus vectorized NTT Phase2 store and INTT Phase1
  load
- `v03_template_config.cu`: v02 plus template-specialized phase config
- `v04_montgomery.cu`: v03 plus Montgomery multiplication and fused input conversion

For each new version, record:

- Goal
- Main implementation change
- Expected bottleneck addressed
- Runtime and GOp/s
- Whether `--verify` passes
