#include <cassert>
#include <cstdio>
#include <cstring>
#include "dsp.h"
#include "inference.h"
#include "simulation_player.h"

int main() {
    sim::Player player;
    int16_t chunk[cfg::chunk];
    assert(player.read(chunk));
    player.command(sim::Command::Pause);
    assert(player.read(chunk));
    uint32_t cursor = player.cursor();
    for (int i = 0; i < 3; ++i) { player.read(chunk); assert(player.cursor() == cursor); }
    for (int16_t x : chunk) assert(x == 0);
    player.command(sim::Command::Pause);
    assert(player.read(chunk));
    assert(player.cursor() == cfg::chunk);
    player.command(sim::Command::Next);
    assert(player.read(chunk) && player.clip() == 1);
    player.command(sim::Command::Restart);
    assert(player.read(chunk) && player.clip() == 1 && player.cursor() == cfg::chunk);

    sim::Player stream;
    static int16_t ring[cfg::samples], window[cfg::samples];
    static float features[cfg::features];
    static Dsp dsp;
    EventGate gate;
    int filled = 0, index = 0, stride = 0, events[6] = {};
    for (int step = 0; step < 6 * 700; ++step) {
        bool reset = stream.read(chunk);
        if (reset) filled = stride = 0;
        for (int16_t x : chunk) { ring[index] = x; index = (index + 1) % cfg::samples; }
        filled += cfg::chunk;
        stride += cfg::chunk;
        if (filled < cfg::samples || stride < cfg::stride) continue;
        stride = 0;
        for (int i = 0; i < cfg::samples; ++i) window[i] = ring[(index+i) % cfg::samples];
        float rms, centroid;
        dsp.extract(window, features, rms, centroid);
        if (gate.update(predict(features), (step+1)*10000LL)) ++events[stream.clip()];
    }
    for (int i = 0; i < 6; ++i) printf("clip=%d events=%d\n", i, events[i]);
    assert(events[0] == 1);
    for (int i = 1; i < 6; ++i) assert(events[i] == 0);
    return 0;
}
