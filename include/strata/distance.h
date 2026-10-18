#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "strata/metric.h"

namespace strata {

// Plain loop versions. They are the reference that faster versions are checked against.
float l2_squared(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product(const float* a, const float* b, std::size_t dim) noexcept;

// Scales v to length 1. A vector of all zeros is left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

// Which set of CPU instructions the distance functions use.
enum class SimdLevel : std::uint8_t {
    Scalar,   // plain loops, works on any CPU
    AutoVec,  // plain loops that the compiler was allowed to turn into AVX2
    Avx2,     // hand-written AVX2 + FMA, 8 floats at a time
    Avx512,   // hand-written AVX-512, 16 floats at a time
};

std::string_view simd_level_name(SimdLevel level) noexcept;
std::optional<SimdLevel> parse_simd_level(std::string_view name) noexcept;

// True if this build contains the level and this CPU can run it.
bool simd_level_supported(SimdLevel level) noexcept;

// All levels this machine can run, slowest first.
std::vector<SimdLevel> supported_simd_levels();

// The level used by distance_function(). Starts as the fastest supported level,
// or the one named in the STRATA_SIMD environment variable (scalar, autovec, avx2, avx512).
SimdLevel active_simd_level() noexcept;

// Changes the active level. Affects indexes created afterwards. Throws if unsupported.
void set_simd_level(SimdLevel level);

using KernelFn = float (*)(const float* a, const float* b, std::size_t dim);

// The raw squared-L2 and dot-product functions for one level. Throws if unsupported.
struct Kernels {
    KernelFn l2_squared;
    KernelFn dot_product;
};
Kernels kernels_for(SimdLevel level);

// Distance between two vectors for a given metric (smaller = closer), using the
// active SIMD level. For Cosine, both vectors must already be normalized.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim);
DistanceFn distance_function(Metric metric) noexcept;

}  // namespace strata
