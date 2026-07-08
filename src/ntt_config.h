#pragma once

#include <cstdint>
#include <vector>

struct ModulusConfig {
    uint32_t qi;
    uint32_t primitive_root;
};

struct VersionConfig {
    int logN;
    std::vector<ModulusConfig> moduli;
};

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
