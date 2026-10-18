// Compiled with -mavx512f. Only <immintrin.h> and <cstddef> are included, so no
// shared inline code gets AVX-512 instructions that other files might pick up.

#include <immintrin.h>

#include <cstddef>

#include "distance_kernels.h"

namespace strata::kernels {
namespace {

// Bit mask with the lowest n bits set (n < 16): selects the leftover floats.
inline __mmask16 tail_mask(std::size_t n) {
    return static_cast<__mmask16>((1u << static_cast<unsigned>(n)) - 1u);
}

}  // namespace

// Two running sums of 16 floats each. The last partial block is read with a
// mask, so there is no separate scalar loop.
float l2_squared_avx512(const float* a, const float* b, std::size_t dim) noexcept {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        const __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        s0 = _mm512_fmadd_ps(d, d, s0);
    }
    if (i < dim) {
        const __mmask16 m = tail_mask(dim - i);
        const __m512 d =
            _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i));
        s1 = _mm512_fmadd_ps(d, d, s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

float dot_product_avx512(const float* a, const float* b, std::size_t dim) noexcept {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    for (; i + 16 <= dim; i += 16) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    }
    if (i < dim) {
        const __mmask16 m = tail_mask(dim - i);
        s1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

}  // namespace strata::kernels
