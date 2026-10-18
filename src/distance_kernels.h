#pragma once

#include <cstddef>

// Distance kernels that live in their own source files, each compiled with
// its own CPU flags. Only call one after checking the CPU supports it.
namespace strata::kernels {

float l2_squared_autovec(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_autovec(const float* a, const float* b, std::size_t dim) noexcept;

float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_avx2(const float* a, const float* b, std::size_t dim) noexcept;

float l2_squared_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_avx512(const float* a, const float* b, std::size_t dim) noexcept;

}  // namespace strata::kernels
