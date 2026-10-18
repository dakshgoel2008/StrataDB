#include "strata/distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
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
