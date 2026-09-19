#pragma once
#include <cstdint>
#include "config.h"
#ifdef SIMULATOR_DEMO
#include "demo_audio.h"
#else
#include "../generated/wokwi/audio_data.h"
#endif

namespace sim {
enum class Command : uint8_t { Next, Restart, Pause };
inline int clipCount() {
#ifdef SIMULATOR_DEMO
    return demo::clipCount;
#else
    return SIM_CLIP_COUNT;
#endif
}
inline uint32_t clipSamples(int clip) {
#ifdef SIMULATOR_DEMO
    (void)clip;
    return cfg::samples;
#else
    return SIM_CLIP_LENGTHS[clip];
#endif
}
inline const char* clipName(int clip) {
#ifdef SIMULATOR_DEMO
    return demo::names[clip];
#else
    return SIM_CLIP_NAMES[clip];
#endif
}
inline int16_t sample(int clip, uint32_t position) {
#ifdef SIMULATOR_DEMO
    return demo::sample(clip, position);
#else
    return SIM_PCM[SIM_CLIP_OFFSETS[clip] + position];
#endif
}

// Proprietário exclusivo: tarefa de captura. Controles chegam por fila FreeRTOS.
class Player {
    int current = 0;
    uint32_t position = 0;
    bool stopped = false, restartPending = true;
public:
    int clip() const { return current; }
    bool paused() const { return stopped; }
    uint32_t cursor() const { return position; }
    void command(Command command) {
        if (command == Command::Next) current = (current+1) % clipCount();
        if (command == Command::Pause) stopped = !stopped;
        else position = 0;
        // Ao retomar começa do início: nunca cortar uma palavra no meio.
        if (command == Command::Pause && !stopped) position = 0;
        restartPending = true;
    }
    bool read(int16_t* pcm) {
        constexpr uint32_t pre = 2 * cfg::sampleRate, post = 3 * cfg::sampleRate;
        if (!stopped && position >= pre + clipSamples(current) + post) {
            current = (current+1) % clipCount();
            position = 0;
            restartPending = true;
        }
        bool restarted = restartPending;
        restartPending = false;
        for (int i = 0; i < cfg::chunk; ++i) {
            pcm[i] = (!stopped && position >= pre && position < pre + clipSamples(current))
                ? sample(current, position-pre) : 0;
            if (!stopped) ++position;
        }
        return restarted;
    }
};
}
