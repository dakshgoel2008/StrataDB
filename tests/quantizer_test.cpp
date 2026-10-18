#include "strata/quantizer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "strata/distance.h"

using strata::ProductQuantizer;
using strata::ScalarQuantizer;

namespace {

std::vector<float> random_vectors(std::size_t n, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (float& x : v) {
        x = dist(rng);
    }
    return v;
}

double slow_l2(const float* a, const float* b, std::size_t dim) {
    double s = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return s;
}

double slow_dot(const float* a, const float* b, std::size_t dim) {
    double s = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

}  // namespace

TEST(ScalarQuantizer, RoundTripErrorIsAtMostHalfAStep) {
    constexpr std::size_t kDim = 32;
    const auto data = random_vectors(500, kDim, 1);
    const auto sq = ScalarQuantizer::train(data, kDim);
    EXPECT_EQ(sq.code_size(), kDim);

    std::vector<std::uint8_t> code(kDim);
    std::vector<float> back(kDim);
    for (std::size_t r = 0; r < 500; ++r) {
        sq.encode(data.data() + r * kDim, code.data());
        sq.decode(code.data(), back.data());
        for (std::size_t i = 0; i < kDim; ++i) {
            ASSERT_LE(std::abs(back[i] - data[r * kDim + i]), sq.scales()[i] * 0.5f + 1e-6f);
        }
    }
}

TEST(ScalarQuantizer, ValuesOutsideTheRangeAreClamped) {
    const std::vector<float> sample = {0, 0, 1, 1};  // two rows of dim 2, range [0, 1]
    const auto sq = ScalarQuantizer::train(sample, 2);
    std::vector<std::uint8_t> code(2);
    const std::vector<float> outside = {-5, 7};
    sq.encode(outside.data(), code.data());
    EXPECT_EQ(code[0], 0);
    EXPECT_EQ(code[1], 255);
}

TEST(ScalarQuantizer, ConstantDimensionIsHandled) {
    const std::vector<float> sample = {3, 1, 3, 2, 3, 5};  // first dimension is always 3
    const auto sq = ScalarQuantizer::train(sample, 2);
    std::vector<std::uint8_t> code(2);
    std::vector<float> back(2);
    const std::vector<float> v = {3, 2};
    sq.encode(v.data(), code.data());
    sq.decode(code.data(), back.data());
    EXPECT_FLOAT_EQ(back[0], 3.0f);
}

TEST(ScalarQuantizer, PreparedDistancesMatchDecodedVectors) {
    constexpr std::size_t kDim = 40;
    const auto data = random_vectors(300, kDim, 2);
    const auto queries = random_vectors(20, kDim, 3);
    const auto sq = ScalarQuantizer::train(data, kDim);

    std::vector<std::uint8_t> code(kDim);
    std::vector<float> decoded(kDim);
    ScalarQuantizer::PreparedQuery pl2;
    ScalarQuantizer::PreparedQuery pdot;
    for (auto level : strata::supported_simd_levels()) {
        const auto k = strata::kernels_for(level);
        for (std::size_t q = 0; q < 20; ++q) {
            const float* query = queries.data() + q * kDim;
            sq.prepare_l2(query, pl2);
            sq.prepare_dot(query, pdot);
            for (std::size_t r = 0; r < 300; r += 7) {
                sq.encode(data.data() + r * kDim, code.data());
                sq.decode(code.data(), decoded.data());
                const double l2 =
                    pl2.constant + k.sq8_l2(pl2.value.data(), pl2.weight.data(), code.data(), kDim);
                const double dot = pdot.constant + k.sq8_dot(pdot.value.data(), code.data(), kDim);
                EXPECT_NEAR(l2, slow_l2(query, decoded.data(), kDim), 1e-3)
                    << simd_level_name(level);
                EXPECT_NEAR(dot, slow_dot(query, decoded.data(), kDim), 1e-3)
                    << simd_level_name(level);
            }
        }
    }
}

TEST(ScalarQuantizer, RejectsBadInput) {
    EXPECT_THROW(ScalarQuantizer::train({}, 4), std::invalid_argument);
    EXPECT_THROW(ScalarQuantizer::train(std::vector<float>(5), 4), std::invalid_argument);
}

TEST(KMeans, FindsWellSeparatedClusters) {
    // Four tight groups of points around far-apart centers.
    const std::vector<std::pair<float, float>> centers = {{0, 0}, {100, 0}, {0, 100}, {100, 100}};
    std::mt19937 rng(4);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::vector<float> points;
    for (int i = 0; i < 400; ++i) {
        const auto& c = centers[static_cast<std::size_t>(i % 4)];
        points.push_back(c.first + noise(rng));
        points.push_back(c.second + noise(rng));
    }
    const auto result = strata::kmeans(points.data(), 400, 2, 4, 20, 9);
    for (const auto& c : centers) {
        bool found = false;
        for (std::size_t j = 0; j < 4; ++j) {
            found |=
                std::abs(result[j * 2] - c.first) < 2 && std::abs(result[j * 2 + 1] - c.second) < 2;
        }
        EXPECT_TRUE(found) << c.first << "," << c.second;
    }
}

TEST(KMeans, SameSeedSameResult) {
    const auto points = random_vectors(1000, 4, 5);
    EXPECT_EQ(strata::kmeans(points.data(), 1000, 4, 16, 10, 3),
              strata::kmeans(points.data(), 1000, 4, 16, 10, 3));
}

TEST(KMeans, MorePointsThanCentroidsNeeded) {
    const auto points = random_vectors(10, 2, 6);
    EXPECT_THROW(strata::kmeans(points.data(), 10, 2, 16, 10, 1), std::invalid_argument);
}

TEST(ProductQuantizer, TableSumEqualsDistanceToDecodedVector) {
    constexpr std::size_t kDim = 32;
    const auto data = random_vectors(2000, kDim, 7);
    const auto queries = random_vectors(10, kDim, 8);
    const auto pq = ProductQuantizer::train(data, kDim, {8, 10, 1, 0});
    EXPECT_EQ(pq.code_size(), 8u);
    EXPECT_EQ(pq.sub_dim(), 4u);

    std::vector<std::uint8_t> code(8);
    std::vector<float> decoded(kDim);
    std::vector<float> l2_table(8 * ProductQuantizer::kCentroids);
    std::vector<float> dot_table(8 * ProductQuantizer::kCentroids);
    for (std::size_t q = 0; q < 10; ++q) {
        const float* query = queries.data() + q * kDim;
        pq.compute_table(query, false, l2_table.data());
        pq.compute_table(query, true, dot_table.data());
        for (std::size_t r = 0; r < 2000; r += 97) {
            pq.encode(data.data() + r * kDim, code.data());
            pq.decode(code.data(), decoded.data());
            EXPECT_NEAR(strata::pq_table_sum(l2_table.data(), code.data(), 8),
                        slow_l2(query, decoded.data(), kDim), 1e-3);
            EXPECT_NEAR(strata::pq_table_sum(dot_table.data(), code.data(), 8),
                        slow_dot(query, decoded.data(), kDim), 1e-3);
        }
    }
}

TEST(ProductQuantizer, MorePiecesMeansSmallerError) {
    constexpr std::size_t kDim = 32;
    const auto data = random_vectors(3000, kDim, 9);
    double previous = 1e30;
    for (std::size_t m : {2u, 8u, 32u}) {
        const auto pq = ProductQuantizer::train(data, kDim, {m, 10, 1, 0});
        std::vector<std::uint8_t> code(m);
        std::vector<float> decoded(kDim);
        double error = 0;
        for (std::size_t r = 0; r < 3000; ++r) {
            pq.encode(data.data() + r * kDim, code.data());
            pq.decode(code.data(), decoded.data());
            error += slow_l2(data.data() + r * kDim, decoded.data(), kDim);
        }
        EXPECT_LT(error, previous) << "m = " << m;
        previous = error;
    }
}

TEST(ProductQuantizer, ResultDoesNotDependOnThreadCount) {
    constexpr std::size_t kDim = 16;
    const auto data = random_vectors(1000, kDim, 10);
    const auto a = ProductQuantizer::train(data, kDim, {4, 8, 2, 1});
    const auto b = ProductQuantizer::train(data, kDim, {4, 8, 2, 8});
    for (std::size_t piece = 0; piece < 4; ++piece) {
        for (std::size_t i = 0; i < ProductQuantizer::kCentroids * 4; ++i) {
            ASSERT_EQ(a.centroids(piece)[i], b.centroids(piece)[i]);
        }
    }
}

TEST(ProductQuantizer, RejectsBadInput) {
    const auto data = random_vectors(300, 10, 11);
    EXPECT_THROW(ProductQuantizer::train(data, 10, {3, 5, 1, 0}),
                 std::invalid_argument);  // 3 doesn't divide 10
    EXPECT_THROW(ProductQuantizer::train(random_vectors(100, 10, 12), 10, {5, 5, 1, 0}),
                 std::invalid_argument);  // fewer than 256 vectors
}
