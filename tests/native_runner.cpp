#include <cstdio>
#include "dsp.h"
#include "inference.h"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    static int16_t audio[cfg::samples];
    FILE* input = fopen(argv[1], "rb");
    if (!input) return 3;
    size_t count = fread(audio, sizeof(int16_t), cfg::samples, input);
    fclose(input);
    if (count != cfg::samples) return 4;
    static Dsp dsp;
    static float features[cfg::features + 3];
    dsp.extract(audio, features, features[cfg::features], features[cfg::features+1]);
    features[cfg::features+2] = predict(features);
    FILE* output = fopen(argv[2], "wb");
    if (!output) return 5;
    fwrite(features, sizeof(float), cfg::features+3, output);
    fclose(output);
    return 0;
}
