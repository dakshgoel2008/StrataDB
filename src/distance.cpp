#include "strata/distance.h"

#include <cmath>

namespace strata {
namespace {

float inner_product_distance(const float* a, const float* b, std::size_t dim) {
    return -dot_product(a, b, dim);
}

float cosine_distance(const float* a, const float* b, std::size_t dim) {
    return 1.0f - dot_product(a, b, dim);
}

}  // namespace

// Four running sums instead of one, so each add doesn't have to wait
// for the previous one to finish.
float l2_squared(const float* a, const float* b, std::size_t dim) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float d0 = a[i] - b[i];
        const float d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2];
        const float d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        s0 += d * d;
    }
    return (s0 + s1) + (s2 + s3);
}

float dot_product(const float* a, const float* b, std::size_t dim) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < dim; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

void normalize(float* v, std::size_t dim) noexcept {
    // Add up in double so long vectors don't lose precision.
    double squares = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        squares += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    if (squares == 0) {
        return;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(squares));
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] *= scale;
    }
}

DistanceFn distance_function(Metric metric) noexcept {
    switch (metric) {
        case Metric::L2:
            return l2_squared;
        case Metric::InnerProduct:
            return inner_product_distance;
        case Metric::Cosine:
            return cosine_distance;
    }
    return l2_squared;
}

}  // namespace strata
