#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include "config.h"
#include "dsp_tables.h"

// Scratch pertence exclusivamente à tarefa de features; sem alocação no loop.
class Dsp {
    float re[cfg::fft], im[cfg::fft], power[cfg::fft / 2 + 1];
    void fft() {
        for (int i = 1, j = 0; i < cfg::fft; ++i) {
            int bit = cfg::fft >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; }
        }
        for (int len = 2; len <= cfg::fft; len <<= 1) {
            const float angle = -6.283185307179586f / len;
            const float wr = cosf(angle), wi = sinf(angle);
            for (int i = 0; i < cfg::fft; i += len) {
                float ur = 1, ui = 0;
                for (int j = 0; j < len / 2; ++j) {
                    int a = i+j, b = a+len/2;
                    float vr = re[b]*ur-im[b]*ui, vi = re[b]*ui+im[b]*ur;
                    re[b] = re[a]-vr; im[b] = im[a]-vi;
                    re[a] += vr; im[a] += vi;
                    float next = ur*wr-ui*wi;
                    ui = ur*wi+ui*wr; ur = next;
                }
            }
        }
    }
public:
    void extract(const int16_t* audio, float* output, float& rms, float& centroid) {
        memset(output, 0, cfg::features * sizeof(float));
        double energy = 0;
        for (int i = 0; i < cfg::samples; ++i) {
            float x = audio[i] / 32768.0f;
            energy += x*x;
        }
        rms = sqrt(energy / cfg::samples);
        centroid = 0;
        for (int bin = 0; bin < cfg::timeBins; ++bin) {
            int first = bin*cfg::frames/cfg::timeBins;
            int last = (bin+1)*cfg::frames/cfg::timeBins;
            for (int frame = first; frame < last; ++frame) {
                const int16_t* input = audio + frame*cfg::chunk;
                float mean = 0;
                for (int i = 0; i < cfg::frame; ++i) mean += input[i]/32768.0f;
                mean /= cfg::frame;
                memset(re, 0, sizeof(re)); memset(im, 0, sizeof(im));
                for (int i = 0; i < cfg::frame; ++i) re[i] = (input[i]/32768.0f-mean)*DSP_HANN[i];
                fft();
                float total = 0, weighted = 0;
                for (int i = 0; i <= cfg::fft/2; ++i) {
                    power[i] = (re[i]*re[i]+im[i]*im[i])/cfg::fft;
                    total += power[i]; weighted += power[i]*i*cfg::sampleRate/cfg::fft;
                }
                centroid += weighted/fmaxf(total, 1e-10f)/cfg::frames;
                int offset = 0;
                for (int m = 0; m < cfg::mels; ++m) {
                    float e = 0;
                    for (int k = 0; k < DSP_MEL_LEN[m]; ++k)
                        e += power[DSP_MEL_START[m]+k]*DSP_MEL_WEIGHTS[offset++];
                    output[bin*cfg::mels+m] += logf(fmaxf(e, 1e-10f))/(last-first);
                }
            }
        }
    }
};
