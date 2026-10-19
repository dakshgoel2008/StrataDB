#include "strata/crc32c.h"

#include <array>
#include <cstring>

namespace strata {

#ifdef STRATA_X86_KERNELS
std::uint32_t crc32c_sse42(const void* data, std::size_t size, std::uint32_t crc) noexcept;
#endif

namespace {

constexpr std::uint32_t kPolynomial = 0x82f63b78;  // bit-reversed 0x1EDC6F41

// tables[0] is the usual one-byte table. tables[k][b] is the CRC of byte b
// followed by k zero bytes, which lets the loop handle 8 bytes per step.
constexpr std::array<std::array<std::uint32_t, 256>, 8> make_tables() {
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    for (std::uint32_t b = 0; b < 256; ++b) {
        std::uint32_t c = b;
        for (int bit = 0; bit < 8; ++bit) {
            c = (c & 1) != 0 ? (c >> 1) ^ kPolynomial : c >> 1;
        }
        t[0][b] = c;
    }
    for (std::size_t k = 1; k < 8; ++k) {
        for (std::size_t b = 0; b < 256; ++b) {
            const std::uint32_t prev = t[k - 1][b];
            t[k][b] = (prev >> 8) ^ t[0][prev & 0xff];
        }
    }
    return t;
}

constexpr auto kTables = make_tables();

using CrcFn = std::uint32_t (*)(const void*, std::size_t, std::uint32_t) noexcept;

CrcFn pick_function() noexcept {
#ifdef STRATA_X86_KERNELS
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse4.2")) {
        return crc32c_sse42;
    }
#endif
    return crc32c_portable;
}

const CrcFn kActive = pick_function();

}  // namespace

std::uint32_t crc32c_portable(const void* data, std::size_t size, std::uint32_t crc) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint32_t c = ~crc;
    while (size >= 8) {
        std::uint32_t lo;
        std::uint32_t hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + 4, 4);
        lo ^= c;
        c = kTables[7][lo & 0xff] ^ kTables[6][(lo >> 8) & 0xff] ^ kTables[5][(lo >> 16) & 0xff] ^
            kTables[4][lo >> 24] ^ kTables[3][hi & 0xff] ^ kTables[2][(hi >> 8) & 0xff] ^
            kTables[1][(hi >> 16) & 0xff] ^ kTables[0][hi >> 24];
        p += 8;
        size -= 8;
    }
    while (size-- > 0) {
        c = (c >> 8) ^ kTables[0][(c ^ *p++) & 0xff];
    }
    return ~c;
}

std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc) noexcept {
    return kActive(data, size, crc);
}

bool crc32c_uses_cpu_instruction() noexcept { return kActive != crc32c_portable; }

}  // namespace strata
