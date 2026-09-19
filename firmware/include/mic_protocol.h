#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace mic {
constexpr size_t samples = 1600;
constexpr size_t payloadBytes = samples * 2;
constexpr uint8_t magic[] = {'M', 'I', 'C', '6', '7', 'P', 'C', 'M'};
inline uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
inline uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t crc = 0xffffffff;
    while (n--) {
        crc ^= *p++;
        for (int i = 0; i < 8; ++i) crc = (crc >> 1) ^ (0xedb88320u & (0u-(crc & 1u)));
    }
    return ~crc;
}
class Parser {
    uint8_t body[4 + payloadBytes + 4];
    size_t matched = 0, used = 0;
public:
    void reset() { matched = used = 0; }
    bool partial() const { return matched != 0; }
    // 0: incomplete, 1: complete and valid, -1: corrupt.
    int feed(uint8_t byte) {
        if (matched < sizeof(magic)) {
            matched = byte == magic[matched] ? matched + 1 : (byte == magic[0] ? 1 : 0);
            return 0;
        }
        body[used++] = byte;
        if (used != sizeof(body)) return 0;
        reset();
        return crc32(body, 4 + payloadBytes) == u32(body + 4 + payloadBytes) ? 1 : -1;
    }
    uint32_t sequence() const { return u32(body); }
    void copy(size_t offset, int16_t* out, size_t count) const {
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = body + 4 + (offset + i) * 2;
            out[i] = static_cast<int16_t>(uint16_t(p[0]) | uint16_t(p[1]) << 8);
        }
    }
};
}
