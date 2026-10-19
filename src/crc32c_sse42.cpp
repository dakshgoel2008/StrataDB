// Compiled with -msse4.2. Only called on CPUs that have the crc32 instruction.

#include <nmmintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata {

std::uint32_t crc32c_sse42(const void* data, std::size_t size, std::uint32_t crc) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t c = ~crc;
    while (size >= 8) {
        std::uint64_t word;
        std::memcpy(&word, p, 8);
        c = _mm_crc32_u64(c, word);
        p += 8;
        size -= 8;
    }
    auto c32 = static_cast<std::uint32_t>(c);
    while (size-- > 0) {
        c32 = _mm_crc32_u8(c32, *p++);
    }
    return ~c32;
}

}  // namespace strata
