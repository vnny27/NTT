#pragma once

#include <cstdint>
#include <vector>

struct ModulusConfig {
    uint32_t qi;
    uint32_t primitive_root;
};

// log2
struct PhaseConfig {
    int radix_stages;      // log2(radix) handled by this phase
    int stage_merging;     // radix-2 stages fused in one local step
    int warp_batching;     // coalescing/batching factor for phase indexing
};

struct TransformConfig {
    PhaseConfig ntt_phase1;
    PhaseConfig ntt_phase2;
    PhaseConfig intt_phase1;
    PhaseConfig intt_phase2;
};

struct VersionConfig {
    int logN;
    std::vector<ModulusConfig> moduli;
    TransformConfig transform;
};

template <int RadixStages, int StageMerging, int WarpBatching>
struct PhaseLaunchConfig {
    static constexpr int radix_stages = RadixStages;
    static constexpr int stage_merging = StageMerging;
    static constexpr int warp_batching = WarpBatching;

    static constexpr PhaseConfig runtime() {
        return {RadixStages, StageMerging, WarpBatching};
    }
};

template <int LogN, typename NttPhase1Config, typename NttPhase2Config,
          typename InttPhase1Config, typename InttPhase2Config>
struct NTTLaunchConfig {
    static constexpr int logN = LogN;

    using NttPhase1 = NttPhase1Config;
    using NttPhase2 = NttPhase2Config;
    using InttPhase1 = InttPhase1Config;
    using InttPhase2 = InttPhase2Config;

    static constexpr TransformConfig runtime_transform() {
        return {
            NttPhase1::runtime(),
            NttPhase2::runtime(),
            InttPhase1::runtime(),
            InttPhase2::runtime(),
        };
    }
};

template <int LogN>
struct DefaultNTTLaunchConfig : NTTLaunchConfig<
    LogN,
    PhaseLaunchConfig<(LogN == 16 ? 7 : LogN - 9),
                      (LogN == 16 ? 4 : 3), 4>,
    PhaseLaunchConfig<9, 3, 0>,
    PhaseLaunchConfig<9, 3, 0>,
    PhaseLaunchConfig<(LogN == 16 ? 7 : LogN - 9),
                      (LogN == 16 ? 4 : 3), 4>
> {
    static_assert(LogN >= 10, "default launch config requires logN >= 10");
};

inline std::vector<ModulusConfig> default_moduli() {
    return {
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
    };
}

template <typename LaunchConfig>
inline VersionConfig make_version_config() {
    return {
        LaunchConfig::logN,
        default_moduli(),
        LaunchConfig::runtime_transform(),
    };
}

inline VersionConfig default_ntt_config() {
    return make_version_config<DefaultNTTLaunchConfig<16>>();
}
