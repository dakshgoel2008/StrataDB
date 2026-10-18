#include "strata/distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// Straightforward double-precision versions to compare against.
double slow_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return s;
}

double slow_dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

std::vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (float& x : v) {
        x = dist(rng);
    }
    return v;
}

}  // namespace

TEST(Distance, SmallKnownValues) {
    const std::vector<float> a = {1, 2, 3};
    const std::vector<float> b = {4, 6, 3};
    EXPECT_FLOAT_EQ(strata::l2_squared(a.data(), b.data(), 3), 25.0f);   // 9 + 16 + 0
    EXPECT_FLOAT_EQ(strata::dot_product(a.data(), b.data(), 3), 25.0f);  // 4 + 12 + 9
}

TEST(Distance, MatchesSlowVersionForManySizes) {
    // Sizes that are and aren't multiples of 4 cover the leftover loop.
    std::mt19937 rng(42);
    for (std::size_t dim : {1u, 2u, 3u, 4u, 5u, 7u, 16u, 100u, 128u, 768u, 960u}) {
        const auto a = random_vector(dim, rng);
        const auto b = random_vector(dim, rng);
        const double tol = 1e-4 * static_cast<double>(dim);
        EXPECT_NEAR(strata::l2_squared(a.data(), b.data(), dim), slow_l2(a, b), tol) << dim;
        EXPECT_NEAR(strata::dot_product(a.data(), b.data(), dim), slow_dot(a, b), tol) << dim;
    }
}

TEST(Distance, NormalizeGivesLengthOne) {
    std::vector<float> v = {3, 4};
    strata::normalize(v.data(), v.size());
    EXPECT_FLOAT_EQ(v[0], 0.6f);
    EXPECT_FLOAT_EQ(v[1], 0.8f);
}

TEST(Distance, NormalizeLeavesZeroVectorAlone) {
    std::vector<float> v = {0, 0, 0};
    strata::normalize(v.data(), v.size());
    EXPECT_EQ(v, (std::vector<float>{0, 0, 0}));
}

TEST(Distance, MetricFunctions) {
    const std::vector<float> a = {1, 0};
    const std::vector<float> same = {1, 0};
    const std::vector<float> opposite = {-1, 0};
    const std::vector<float> sideways = {0, 1};

    const auto cosine = strata::distance_function(strata::Metric::Cosine);
    EXPECT_FLOAT_EQ(cosine(a.data(), same.data(), 2), 0.0f);
    EXPECT_FLOAT_EQ(cosine(a.data(), sideways.data(), 2), 1.0f);
    EXPECT_FLOAT_EQ(cosine(a.data(), opposite.data(), 2), 2.0f);

    // Inner product: a bigger dot product means closer, so distance is its negative.
    const auto ip = strata::distance_function(strata::Metric::InnerProduct);
    EXPECT_FLOAT_EQ(ip(a.data(), same.data(), 2), -1.0f);

    const auto l2 = strata::distance_function(strata::Metric::L2);
    EXPECT_FLOAT_EQ(l2(a.data(), opposite.data(), 2), 4.0f);
}

TEST(Metric, NamesRoundTrip) {
    for (auto m : {strata::Metric::L2, strata::Metric::InnerProduct, strata::Metric::Cosine}) {
        EXPECT_EQ(strata::parse_metric(strata::metric_name(m)), m);
    }
    EXPECT_EQ(strata::parse_metric("inner_product"), strata::Metric::InnerProduct);
    EXPECT_FALSE(strata::parse_metric("manhattan").has_value());
}

namespace {

// Allowed difference from the double-precision answer. Different summation order
// changes rounding, so the error is measured against the size of the terms added.
double tolerance(const std::vector<float>& a, const std::vector<float>& b, std::size_t offset,
                 std::size_t dim, bool l2) {
    double magnitude = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double x = a[offset + i];
        const double y = b[offset + i];
        magnitude += l2 ? (x - y) * (x - y) : std::abs(x * y);
    }
    return 1e-5 * magnitude + 1e-6;
}

void check_kernels(strata::SimdLevel level) {
    const auto k = strata::kernels_for(level);
    std::mt19937 rng(7);
    std::vector<std::size_t> dims;
    for (std::size_t d = 1; d <= 80; ++d) {
        dims.push_back(d);
    }
    for (std::size_t d : {100u, 127u, 128u, 129u, 768u, 960u, 1000u}) {
        dims.push_back(d);
    }
    for (std::size_t dim : dims) {
        // offset 1 makes the pointers misaligned, like a query in a plain std::vector.
        for (std::size_t offset : {0u, 1u}) {
            const auto a = random_vector(dim + 1, rng);
            const auto b = random_vector(dim + 1, rng);
            const std::vector<float> sa(a.begin() + static_cast<long>(offset),
                                        a.begin() + static_cast<long>(offset + dim));
            const std::vector<float> sb(b.begin() + static_cast<long>(offset),
                                        b.begin() + static_cast<long>(offset + dim));
            EXPECT_NEAR(k.l2_squared(a.data() + offset, b.data() + offset, dim), slow_l2(sa, sb),
                        tolerance(a, b, offset, dim, true))
                << simd_level_name(level) << " l2 dim " << dim;
            EXPECT_NEAR(k.dot_product(a.data() + offset, b.data() + offset, dim), slow_dot(sa, sb),
                        tolerance(a, b, offset, dim, false))
                << simd_level_name(level) << " dot dim " << dim;
        }
    }
}

// Puts the original level back when a test ends.
struct RestoreSimdLevel {
    strata::SimdLevel saved = strata::active_simd_level();
    ~RestoreSimdLevel() { strata::set_simd_level(saved); }
};

}  // namespace

TEST(Simd, ScalarIsAlwaysSupported) {
    EXPECT_TRUE(strata::simd_level_supported(strata::SimdLevel::Scalar));
    EXPECT_EQ(strata::supported_simd_levels().front(), strata::SimdLevel::Scalar);
}

TEST(Simd, EverySupportedLevelMatchesReference) {
    for (auto level : strata::supported_simd_levels()) {
        check_kernels(level);
    }
}

TEST(Simd, Avx512MatchesReference) {
    if (!strata::simd_level_supported(strata::SimdLevel::Avx512)) {
        GTEST_SKIP() << "this CPU has no AVX-512";
    }
    check_kernels(strata::SimdLevel::Avx512);
}

TEST(Simd, UnsupportedLevelIsRejected) {
    for (auto level :
         {strata::SimdLevel::AutoVec, strata::SimdLevel::Avx2, strata::SimdLevel::Avx512}) {
        if (!strata::simd_level_supported(level)) {
            EXPECT_THROW(strata::kernels_for(level), std::invalid_argument);
            EXPECT_THROW(strata::set_simd_level(level), std::invalid_argument);
        }
    }
}

TEST(Simd, SetLevelChangesDistanceFunction) {
    RestoreSimdLevel restore;
    strata::set_simd_level(strata::SimdLevel::Scalar);
    EXPECT_EQ(strata::active_simd_level(), strata::SimdLevel::Scalar);
    EXPECT_EQ(strata::distance_function(strata::Metric::L2), &strata::l2_squared);
    for (auto level : strata::supported_simd_levels()) {
        strata::set_simd_level(level);
        EXPECT_EQ(strata::distance_function(strata::Metric::L2),
                  strata::kernels_for(level).l2_squared);
    }
}

TEST(Simd, NamesRoundTrip) {
    for (auto level : {strata::SimdLevel::Scalar, strata::SimdLevel::AutoVec,
                       strata::SimdLevel::Avx2, strata::SimdLevel::Avx512}) {
        EXPECT_EQ(strata::parse_simd_level(strata::simd_level_name(level)), level);
    }
    EXPECT_FALSE(strata::parse_simd_level("sse").has_value());
}
