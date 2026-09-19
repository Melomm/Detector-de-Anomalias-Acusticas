#include "mic_protocol.h"
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::ifstream input(argv[1], std::ios::binary);
    mic::Parser parser;
    char byte;
    while (input.get(byte)) {
        int result = parser.feed(static_cast<uint8_t>(byte));
        if (result < 0) std::cout << "corrupt\n";
        if (result == 1) {
            int16_t pcm[mic::samples];
            parser.copy(0, pcm, mic::samples);
            std::cout << parser.sequence() << ' ' << pcm[0] << ' ' << pcm[mic::samples-1] << '\n';
        }
    }
}
