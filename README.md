# CUDA NTT Benchmark Notes

This directory tracks CUDA NTT baseline and optimization versions.

## Current Baseline

`src/v00_base_ntt.cu` is the first correctness-oriented GPU baseline.

- Data type: `uint32_t`
- Transform size: `N = 1 << logN`
- Modulus layout: multi-limb RNS-style layout, `A[limb * N + i]`
- Kernel shape: one CUDA kernel launch per NTT/INTT stage
- Verification: CPU butterfly NTT and NTT-to-INTT roundtrip

The current benchmark restores the device input before each measured transform. The reported time includes that device-to-device restore copy.

## Build

From this directory:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 \
  src/v00_base_ntt.cu -o v00_base_ntt
```

With NVTX ranges for Nsight:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -DUSE_NVTX \
  -I /usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3 \
  src/v00_base_ntt.cu -o v00_base_ntt
```

Adjust `sm_89` if the target GPU uses a different architecture.

## Run

Benchmark only:

```bash
./v00_base_ntt
```

Benchmark and verify against the CPU reference:

```bash
./v00_base_ntt --verify
```

Short aliases:

```bash
./v00_base_ntt -v
./v00_base_ntt -h
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

## Profiling

Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx --stats=true \
  -o ntt_v00_nsys ./v00_base_ntt --verify
```

Nsight Compute:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all \
  --nvtx --nvtx-include "custom_ntt/" \
  -o ntt_v00_ncu ./v00_base_ntt
```

## Results Log

The benchmark time currently includes restoring the device input before each transform.
`default` refers to the config in `src/ntt_config.h`.

| Version | Config | Limbs | NTT Time (ms) | NTT GOp/s | INTT Time (ms) | INTT GOp/s | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v00 | default | 16 | 0.078 | 323.995 | 0.089 | 283.712 | Per-stage kernel baseline |
| v01 |  |  |  |  |  |  |  |
| v02 |  |  |  |  |  |  |  |

## Version Notes

Use this pattern when adding later versions:

- `v00_base_ntt.cu`: naive per-stage GPU baseline
- `v01_*.cu`: first optimization
- `v02_*.cu`: next optimization

For each new version, record:

- Goal
- Main implementation change
- Expected bottleneck addressed
- Runtime and GOp/s
- Whether `--verify` passes
