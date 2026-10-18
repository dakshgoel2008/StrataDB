#pragma once

#include <cstddef>

#include "strata/metric.h"

namespace strata {

// Plain loop versions. They are the reference that faster versions are checked against.
float l2_squared(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product(const float* a, const float* b, std::size_t dim) noexcept;

// Scales v to length 1. A vector of all zeros is left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

// Distance between two vectors for a given metric (smaller = closer).
// For Cosine, both vectors must already be normalized.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim);
DistanceFn distance_function(Metric metric) noexcept;

}  // namespace strata
