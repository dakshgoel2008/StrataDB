// Compiled with -mavx2 -mfma -ffast-math. The loops are written plainly; the
// compiler is allowed to reorder the additions, which lets it use AVX2 by itself.
//
// Only <cstddef> is included on purpose: inline code from other headers compiled
// here would contain AVX2 instructions and could be picked up by other files.

#include <cstddef>

#include "distance_kernels.h"

namespace strata::kernels {

float l2_squared_autovec(const float* a, const float* b, std::size_t dim) noexcept {
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float dot_product_autovec(const float* a, const float* b, std::size_t dim) noexcept {
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

}  // namespace strata::kernels
