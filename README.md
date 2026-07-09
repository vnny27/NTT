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

- Data type: `uint32_t`
- Transform size: `N = 1 << logN`
- Modulus layout: multi-limb RNS-style layout, `A[limb * N + i]`
- Verification: CPU butterfly NTT and NTT-to-INTT roundtrip

Kernel shape by version:

- `v00`: one CUDA kernel launch per radix-2 NTT/INTT stage
- `v01`: two phase kernels for NTT/INTT, with stage merging inside each phase

The current benchmark restores the device input before each measured transform.
The reported time includes that device-to-device restore copy.

## Build

From this directory:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 \
  src/<version>.cu -o <version>
```

For example, `src/v01_radix_stages.cu` builds to `./v01_radix_stages`.

With NVTX ranges for Nsight:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -DUSE_NVTX \
  -I /usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3 \
  src/<version>.cu -o <version>
```

When `USE_NVTX` is enabled, `BENCHMARK_ITERATIONS` is set to `1` to keep profiling traces compact.

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

`--base` is accepted by the harness, but no external NTT library comparison is implemented yet.

## Config

Edit the shared config in `src/ntt_config.h`:

```cpp
inline VersionConfig default_ntt_config() {
    return {
        16,
        {
            {974258177u, 3u},
            {1081212929u, 6u},
            {1196556289u, 7u},
            {993263617u, 5u},
            {989986817u, 3u},
            {1074266113u, 5u},
            {1168900097u, 5u},
            {1010565121u, 7u},
            {957349889u, 6u},
            {1073872897u, 7u},
            {1209139201u, 31u},
            {994705409u, 3u},
            {998244353u, 3u},
            {1073479681u, 11u},
            {1158676481u, 3u},
            {1016463361u, 38u},
        },
        {
            {7, 4, 4}, // NTT phase 1
            {9, 3, 0}, // NTT phase 2
            {9, 3, 0}, // INTT phase 1
            {7, 4, 4}, // INTT phase 2
        },
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
{radix_stages, stage_merging, log_warp_batching}
```

- `radix_stages`: `log2(radix)` handled by the phase.
- `stage_merging`: radix-2 stages fused inside one local register/shared-memory step.
- `log_warp_batching`: batching factor used by phase indexing to improve coalescing.

The phase config fields are used by phase-based versions such as `v01`.

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
| v00 | default | 16 | 0.078 | 323.995 | 0.089 | 283.712 | One kernel launch per radix-2 stage |
| v01 | default | 16 | 0.0673 | 378.092 | 0.073 | 346.141 | Two-phase radix-staged kernel with stage merging |
| v02 |  |  |  |  |  |  |  |

## Version Notes

Use this pattern when adding later versions:

- `v00_base_ntt.cu`: naive per-stage GPU baseline
- `v01_radix_stages.cu`: phase-split radix-staged kernel
- `v02_*.cu`: next optimization

For each new version, record:

- Goal
- Main implementation change
- Expected bottleneck addressed
- Runtime and GOp/s
- Whether `--verify` passes
