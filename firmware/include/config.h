#pragma once
#include <cstdint>
namespace cfg {
constexpr int sampleRate = 16000;
constexpr int samples = 32000;
constexpr int chunk = 160;
constexpr int stride = 4000;
constexpr int frame = 400;
constexpr int fft = 512;
constexpr int mels = 24;
constexpr int timeBins = 24;
constexpr int features = mels * timeBins;
constexpr int frames = 1 + (samples - frame) / chunk;
constexpr int bclk = 14;
constexpr int ws = 27;
constexpr int din = 32;
constexpr int led = 2;
constexpr int buzzer = -1; // Buzzer ativo via transistor: defina GPIO livre, ex. 21.
constexpr int confirmations = 2;
constexpr int64_t cooldownUs = 1500000;
constexpr int64_t alertUs = 250000;
}
