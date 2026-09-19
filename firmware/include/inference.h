#pragma once
#include "config.h"
#include <cmath>
#if defined(SIMULATOR_DEMO)
#include "demo_audio.h"
#define MODEL_READY 0
static const float MODEL_THRESHOLD = 0.8f;
#elif defined(SIMULATOR_REAL)
#include "../generated/wokwi/model_data.h"
#elif __has_include("model_data.h")
#include "model_data.h"
#else
#ifdef REQUIRE_TRAINED_MODEL
#error "Modelo ausente: execute detector67 train antes de compilar detector. Use diagnostic para testar hardware."
#endif
#define MODEL_READY 0
static const float MODEL_THRESHOLD = 2.0f;
#endif

inline float predict(const float* features) {
#if defined(SIMULATOR_DEMO)
    return demo::score(features);
#elif MODEL_READY
    float hidden[MODEL_HIDDEN];
    for (int j = 0; j < MODEL_HIDDEN; ++j) hidden[j] = MODEL_B1[j];
    for (int i = 0; i < cfg::features; ++i) {
        float x = (features[i]-MODEL_MEAN[i])/MODEL_SCALE[i];
        for (int j = 0; j < MODEL_HIDDEN; ++j) hidden[j] += x*MODEL_W1[i*MODEL_HIDDEN+j];
    }
    float logit = MODEL_B2[0];
    for (int j = 0; j < MODEL_HIDDEN; ++j) logit += fmaxf(0, hidden[j])*MODEL_W2[j];
    return 1.0f/(1.0f+expf(-logit));
#else
    (void)features;
    return 0;
#endif
}

class EventGate {
    int hits = 0, negatives = 0;
    int64_t lastEvent = -1000000000, lastTime = -1000000000;
    bool armed = true;
public:
    bool update(float score, int64_t now) {
        if (now-lastTime > 375000) hits = 0;
        lastTime = now;
        if (score >= MODEL_THRESHOLD) { ++hits; negatives = 0; }
        else { hits = 0; if (++negatives >= 2) armed = true; }
        if (armed && hits >= cfg::confirmations && now-lastEvent >= cfg::cooldownUs) {
            armed = false; lastEvent = now; return true;
        }
        return false;
    }
};
