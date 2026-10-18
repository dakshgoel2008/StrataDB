#include "strata/distance.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "distance_kernels.h"

namespace strata {
namespace {

constexpr std::array<SimdLevel, 4> kAllLevels = {SimdLevel::Scalar, SimdLevel::AutoVec,
                                                 SimdLevel::Avx2, SimdLevel::Avx512};

template <KernelFn Dot>
float inner_product_distance(const float* a, const float* b, std::size_t dim) {
    return -Dot(a, b, dim);
}

template <KernelFn Dot>
float cosine_distance(const float* a, const float* b, std::size_t dim) {
    return 1.0f - Dot(a, b, dim);
}

// Everything distance_function() and kernels_for() can hand out for one SIMD level.
struct LevelFunctions {
    KernelFn l2;
    KernelFn dot;
    DistanceFn inner_product;
    DistanceFn cosine;
    Sq8L2Fn sq8_l2;
    Sq8DotFn sq8_dot;
};

template <KernelFn L2, KernelFn Dot, Sq8L2Fn SqL2, Sq8DotFn SqDot>
constexpr LevelFunctions make_level() {
    return {L2, Dot, inner_product_distance<Dot>, cosine_distance<Dot>, SqL2, SqDot};
}

constexpr LevelFunctions kScalar =
    make_level<l2_squared, dot_product, kernels::sq8_l2_scalar, kernels::sq8_dot_scalar>();
#if defined(STRATA_X86_KERNELS)
// SQ8 has no separate compiler-vectorized or AVX-512 version; those levels reuse others.
constexpr LevelFunctions kAutoVec =
    make_level<kernels::l2_squared_autovec, kernels::dot_product_autovec, kernels::sq8_l2_scalar,
               kernels::sq8_dot_scalar>();
constexpr LevelFunctions kAvx2 = make_level<kernels::l2_squared_avx2, kernels::dot_product_avx2,
                                            kernels::sq8_l2_avx2, kernels::sq8_dot_avx2>();
constexpr LevelFunctions kAvx512 =
    make_level<kernels::l2_squared_avx512, kernels::dot_product_avx512, kernels::sq8_l2_avx2,
               kernels::sq8_dot_avx2>();
#endif

const LevelFunctions& functions(SimdLevel level) noexcept {
#if defined(STRATA_X86_KERNELS)
    switch (level) {
        case SimdLevel::Scalar:
            return kScalar;
        case SimdLevel::AutoVec:
            return kAutoVec;
        case SimdLevel::Avx2:
            return kAvx2;
        case SimdLevel::Avx512:
            return kAvx512;
    }
#endif
    (void)level;
    return kScalar;
}

bool cpu_can_run(SimdLevel level) noexcept {
#if defined(STRATA_X86_KERNELS)
    __builtin_cpu_init();
    switch (level) {
        case SimdLevel::Scalar:
            return true;
        case SimdLevel::AutoVec:
        case SimdLevel::Avx2:
            return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        case SimdLevel::Avx512:
            return __builtin_cpu_supports("avx512f");
    }
    return false;
#else
    return level == SimdLevel::Scalar;
#endif
}

SimdLevel fastest_supported() noexcept {
    for (SimdLevel level : {SimdLevel::Avx512, SimdLevel::Avx2}) {
        if (cpu_can_run(level)) {
            return level;
        }
    }
    return SimdLevel::Scalar;
}

SimdLevel starting_level() noexcept {
    const char* requested = std::getenv("STRATA_SIMD");
    if (requested == nullptr) {
        return fastest_supported();
    }
    const auto level = parse_simd_level(requested);
    if (!level || !cpu_can_run(*level)) {
        std::fprintf(stderr, "strata: STRATA_SIMD=%s is not available here, using %s\n", requested,
                     std::string(simd_level_name(fastest_supported())).c_str());
        return fastest_supported();
    }
    return *level;
}

std::atomic<SimdLevel>& active_level() noexcept {
    static std::atomic<SimdLevel> level{starting_level()};
    return level;
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

namespace kernels {

float sq8_l2_scalar(const float* value, const float* weight, const std::uint8_t* code,
                    std::size_t dim) noexcept {
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = value[i] - static_cast<float>(code[i]);
        sum += weight[i] * d * d;
    }
    return sum;
}

float sq8_dot_scalar(const float* value, const std::uint8_t* code, std::size_t dim) noexcept {
    float sum = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += value[i] * static_cast<float>(code[i]);
    }
    return sum;
}

}  // namespace kernels

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

std::string_view simd_level_name(SimdLevel level) noexcept {
    switch (level) {
        case SimdLevel::Scalar:
            return "scalar";
        case SimdLevel::AutoVec:
            return "autovec";
        case SimdLevel::Avx2:
            return "avx2";
        case SimdLevel::Avx512:
            return "avx512";
    }
    return "unknown";
}

std::optional<SimdLevel> parse_simd_level(std::string_view name) noexcept {
    for (SimdLevel level : kAllLevels) {
        if (simd_level_name(level) == name) {
            return level;
        }
    }
    return std::nullopt;
}

bool simd_level_supported(SimdLevel level) noexcept { return cpu_can_run(level); }

std::vector<SimdLevel> supported_simd_levels() {
    std::vector<SimdLevel> levels;
    for (SimdLevel level : kAllLevels) {
        if (cpu_can_run(level)) {
            levels.push_back(level);
        }
    }
    return levels;
}

SimdLevel active_simd_level() noexcept { return active_level().load(); }

void set_simd_level(SimdLevel level) {
    if (!cpu_can_run(level)) {
        throw std::invalid_argument("SIMD level '" + std::string(simd_level_name(level)) +
                                    "' is not supported on this machine");
    }
    active_level().store(level);
}

Kernels kernels_for(SimdLevel level) {
    if (!cpu_can_run(level)) {
        throw std::invalid_argument("SIMD level '" + std::string(simd_level_name(level)) +
                                    "' is not supported on this machine");
    }
    const LevelFunctions& f = functions(level);
    return {f.l2, f.dot, f.sq8_l2, f.sq8_dot};
}

DistanceFn distance_function(Metric metric) noexcept {
    const LevelFunctions& f = functions(active_simd_level());
    switch (metric) {
        case Metric::L2:
            return f.l2;
        case Metric::InnerProduct:
            return f.inner_product;
        case Metric::Cosine:
            return f.cosine;
    }
    return f.l2;
}

}  // namespace strata
