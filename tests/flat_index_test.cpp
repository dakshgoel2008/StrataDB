#include "strata/flat_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "strata/distance.h"

using strata::FlatIndex;
using strata::Metric;
using strata::RowId;

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

// Sorts every row id by distance, the slowest possible way, as a reference.
std::vector<RowId> slow_top_k(const std::vector<float>& base, const float* query, std::size_t dim,
                              Metric metric, std::size_t k) {
    std::vector<float> q(query, query + dim);
    std::vector<float> data = base;
    const std::size_t n = base.size() / dim;
    if (metric == Metric::Cosine) {
        strata::normalize(q.data(), dim);
        for (std::size_t i = 0; i < n; ++i) {
            strata::normalize(data.data() + i * dim, dim);
        }
    }
    const auto distance = strata::distance_function(metric);
    std::vector<RowId> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](RowId a, RowId b) {
        const auto ua = static_cast<std::size_t>(a);
        const auto ub = static_cast<std::size_t>(b);
        return distance(q.data(), data.data() + ua * dim, dim) <
               distance(q.data(), data.data() + ub * dim, dim);
    });
    order.resize(std::min(k, n));
    return order;
}

struct Results {
    std::vector<RowId> ids;
    std::vector<float> distances;
};

Results search(const FlatIndex& index, const std::vector<float>& query, std::size_t k) {
    Results r{std::vector<RowId>(k), std::vector<float>(k)};
    index.search(query, k, r.ids, r.distances);
    return r;
}

}  // namespace

TEST(FlatIndex, RejectsZeroDim) { EXPECT_THROW(FlatIndex(0, Metric::L2), std::invalid_argument); }

TEST(FlatIndex, EmptyIndexReturnsNoResults) {
    FlatIndex index(4, Metric::L2);
    const auto r = search(index, {1, 2, 3, 4}, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(r.ids[i], strata::kNoRow);
        EXPECT_EQ(r.distances[i], std::numeric_limits<float>::infinity());
    }
}

TEST(FlatIndex, KLargerThanSizePadsTheRest) {
    FlatIndex index(2, Metric::L2);
    index.add(std::vector<float>{0, 0, 10, 10});
    const auto r = search(index, {1, 1}, 4);
    EXPECT_EQ(r.ids, (std::vector<RowId>{0, 1, strata::kNoRow, strata::kNoRow}));
    EXPECT_FLOAT_EQ(r.distances[0], 2.0f);
    EXPECT_FLOAT_EQ(r.distances[1], 162.0f);
}

TEST(FlatIndex, AddAssignsIdsInOrder) {
    FlatIndex index(1, Metric::L2);
    index.add(std::vector<float>{5});
    index.add(std::vector<float>{1, 3});
    EXPECT_EQ(index.size(), 3u);
    const auto r = search(index, {0}, 3);
    EXPECT_EQ(r.ids, (std::vector<RowId>{1, 2, 0}));  // values 1, 3, 5
}

TEST(FlatIndex, OddDimWorksWithRowPadding) {
    // dim 3 is stored in 16-float rows; padding must not change any distance.
    FlatIndex index(3, Metric::L2);
    index.add(std::vector<float>{1, 1, 1, 2, 2, 2});
    const auto r = search(index, {2, 2, 2}, 2);
    EXPECT_EQ(r.ids, (std::vector<RowId>{1, 0}));
    EXPECT_FLOAT_EQ(r.distances[0], 0.0f);
    EXPECT_FLOAT_EQ(r.distances[1], 3.0f);
}

TEST(FlatIndex, DuplicateVectorsAreOrderedById) {
    FlatIndex index(2, Metric::L2);
    index.add(std::vector<float>{1, 1, 1, 1, 1, 1});
    const auto r = search(index, {1, 1}, 3);
    EXPECT_EQ(r.ids, (std::vector<RowId>{0, 1, 2}));
}

TEST(FlatIndex, MatchesSlowSearchForEveryMetric) {
    constexpr std::size_t kDim = 20;
    constexpr std::size_t kN = 500;
    constexpr std::size_t kK = 10;
    const auto base = random_vectors(kN, kDim, 1);
    const auto queries = random_vectors(20, kDim, 2);

    for (auto metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        FlatIndex index(kDim, metric);
        index.add(base);
        for (std::size_t q = 0; q < 20; ++q) {
            const std::vector<float> query(queries.begin() + static_cast<long>(q * kDim),
                                           queries.begin() + static_cast<long>((q + 1) * kDim));
            const auto r = search(index, query, kK);
            EXPECT_EQ(r.ids, slow_top_k(base, query.data(), kDim, metric, kK))
                << strata::metric_name(metric) << " query " << q;
            EXPECT_TRUE(std::is_sorted(r.distances.begin(), r.distances.end()));
        }
    }
}

TEST(FlatIndex, CosineIgnoresVectorLength) {
    FlatIndex index(2, Metric::Cosine);
    index.add(std::vector<float>{10, 0, 0, 0.5f});
    // Same direction as row 1, much longer.
    const auto r = search(index, {0, 100}, 2);
    EXPECT_EQ(r.ids[0], 1);
    EXPECT_NEAR(r.distances[0], 0.0f, 1e-6f);
}

TEST(FlatIndex, BatchMatchesSingleSearchForAnyThreadCount) {
    constexpr std::size_t kDim = 16;
    constexpr std::size_t kK = 5;
    constexpr std::size_t kQueries = 37;  // not a multiple of the batch size
    const auto base = random_vectors(300, kDim, 3);
    const auto queries = random_vectors(kQueries, kDim, 4);
    FlatIndex index(kDim, Metric::L2);
    index.add(base);

    std::vector<RowId> expected(kQueries * kK);
    std::vector<float> expected_dist(kQueries * kK);
    for (std::size_t q = 0; q < kQueries; ++q) {
        index.search(std::span(queries).subspan(q * kDim, kDim), kK,
                     std::span(expected).subspan(q * kK, kK),
                     std::span(expected_dist).subspan(q * kK, kK));
    }

    for (std::size_t threads : {1u, 2u, 7u, 0u}) {
        std::vector<RowId> ids(kQueries * kK);
        std::vector<float> dist(kQueries * kK);
        index.search_batch(queries, kK, ids, dist, threads);
        EXPECT_EQ(ids, expected) << threads << " threads";
        EXPECT_EQ(dist, expected_dist) << threads << " threads";
    }
}

TEST(FlatIndex, RejectsWrongSizes) {
    FlatIndex index(4, Metric::L2);
    EXPECT_THROW(index.add(std::vector<float>{1, 2, 3}), std::invalid_argument);

    std::vector<RowId> ids(2);
    std::vector<float> dist(2);
    EXPECT_THROW(index.search(std::vector<float>{1, 2}, 2, ids, dist), std::invalid_argument);
    EXPECT_THROW(index.search(std::vector<float>{1, 2, 3, 4}, 3, ids, dist), std::invalid_argument);
    EXPECT_THROW(index.search_batch(std::vector<float>(8), 2, ids, dist), std::invalid_argument);
}
