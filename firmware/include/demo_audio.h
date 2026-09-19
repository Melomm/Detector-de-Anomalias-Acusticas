#pragma once
#include "config.h"
#include <cmath>
#include <cstdint>

// Demonstração de tons. Não é reconhecimento de fala nem modelo treinado.
namespace demo {
constexpr int clipCount = 6;
static const char* const names[] = {
    "demo_par_440_1100", "demo_440_isolado", "demo_1100_isolado",
    "demo_ordem_inversa", "demo_ruido", "demo_silencio"
};
inline int16_t sample(int clip, uint32_t position) {
    if (clip == 5) return 0;
    if (clip == 4) {
        uint32_t value = position * 747796405u + 2891336453u;
        value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
        return static_cast<int16_t>(((value >> 22u) ^ value) & 8191u) - 4096;
    }
    float t = position / float(cfg::sampleRate);
    float frequency = 0, local = 0;
    if (t >= 0.35f && t < 0.85f && clip != 2) {
        frequency = clip == 3 ? 1100 : 440;
        local = t - 0.35f;
    }
    if (t >= 1.05f && t < 1.55f && clip != 1) {
        frequency = clip == 3 ? 440 : 1100;
        local = t - 1.05f;
    }
    float envelope = fminf(1, fminf(local / 0.02f, (0.5f-local) / 0.02f));
    return frequency ? static_cast<int16_t>(6000 * envelope * sinf(6.28318530718f * frequency * t)) : 0;
}

inline float score(const float* features) {
    int lowCount = 0, highCount = 0, lastLow = -1, firstHigh = cfg::timeBins;
    for (int t = 0; t < cfg::timeBins; ++t) {
        float low = features[t * cfg::mels + 3];
        float high = features[t * cfg::mels + 8];
        if (low > -5 && low-high > 4) { ++lowCount; lastLow = t; }
        if (high > -5 && high-low > 4) {
            ++highCount;
            if (firstHigh == cfg::timeBins) firstHigh = t;
        }
    }
    return lowCount >= 3 && highCount >= 3 && lastLow < firstHigh ? 0.95f : 0.05f;
}
}
