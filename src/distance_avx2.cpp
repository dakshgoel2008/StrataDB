// Compiled with -mavx2 -mfma. Only <immintrin.h> and plain C headers are included, so
// no shared inline code gets AVX2 instructions that other files might pick up.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "distance_kernels.h"

namespace strata::kernels {
namespace {

// Adds up the 8 floats in a register.
inline float sum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

}  // namespace

// Four running sums of 8 floats each: 32 floats per loop, and the four
// multiply-adds don't wait on each other.
float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    float total = sum8(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float dot_product_avx2(const float* a, const float* b, std::size_t dim) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= dim; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    }
    float total = sum8(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

namespace {

// Loads 8 code bytes and turns them into 8 floats.
inline __m256 load_codes(const std::uint8_t* code) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code));
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
}

}  // namespace

float sq8_l2_avx2(const float* value, const float* weight, const std::uint8_t* code,
                  std::size_t dim) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(value + i), load_codes(code + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(value + i + 8), load_codes(code + i + 8));
        s0 = _mm256_fmadd_ps(_mm256_mul_ps(d0, _mm256_loadu_ps(weight + i)), d0, s0);
        s1 = _mm256_fmadd_ps(_mm256_mul_ps(d1, _mm256_loadu_ps(weight + i + 8)), d1, s1);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(value + i), load_codes(code + i));
        s0 = _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_loadu_ps(weight + i)), d, s0);
    }
    float total = sum8(_mm256_add_ps(s0, s1));
    for (; i < dim; ++i) {
        const float d = value[i] - static_cast<float>(code[i]);
        total += weight[i] * d * d;
    }
    return total;
}

float sq8_dot_avx2(const float* value, const std::uint8_t* code, std::size_t dim) noexcept {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(value + i), load_codes(code + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(value + i + 8), load_codes(code + i + 8), s1);
    }
    for (; i + 8 <= dim; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(value + i), load_codes(code + i), s0);
    }
    float total = sum8(_mm256_add_ps(s0, s1));
    for (; i < dim; ++i) {
        total += value[i] * static_cast<float>(code[i]);
    }
    return total;
}

}  // namespace strata::kernels
