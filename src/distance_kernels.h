#pragma once

#include <cstddef>
#include <cstdint>

// Distance kernels that live in their own source files, each compiled with
// its own CPU flags. Only call one after checking the CPU supports it.
namespace strata::kernels {

float sq8_l2_scalar(const float* value, const float* weight, const std::uint8_t* code,
                    std::size_t dim) noexcept;
float sq8_dot_scalar(const float* value, const std::uint8_t* code, std::size_t dim) noexcept;

float l2_squared_autovec(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_autovec(const float* a, const float* b, std::size_t dim) noexcept;

float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_l2_avx2(const float* value, const float* weight, const std::uint8_t* code,
                  std::size_t dim) noexcept;
float sq8_dot_avx2(const float* value, const std::uint8_t* code, std::size_t dim) noexcept;

float l2_squared_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float dot_product_avx512(const float* a, const float* b, std::size_t dim) noexcept;

}  // namespace strata::kernels
