#include "strata/hnsw_index.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "strata/flat_index.h"

using strata::FlatIndex;
using strata::HnswIndex;
using strata::HnswParams;
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

std::vector<float> row(const std::vector<float>& data, std::size_t i, std::size_t dim) {
    return {data.begin() + static_cast<long>(i * dim),
            data.begin() + static_cast<long>((i + 1) * dim)};
}

// Share of the exact top-k (from FlatIndex) that HNSW also finds.
double recall(const HnswIndex& hnsw, const std::vector<float>& base,
              const std::vector<float>& queries, std::size_t k, std::size_t ef) {
    const std::size_t dim = hnsw.dim();
    FlatIndex flat(dim, hnsw.metric());
    flat.add(base);
    const std::size_t nq = queries.size() / dim;

    std::vector<RowId> exact(nq * k), approx(nq * k);
    std::vector<float> d1(nq * k), d2(nq * k);
    flat.search_batch(queries, k, exact, d1);
    hnsw.search_batch(queries, k, ef, approx, d2);

    std::size_t hits = 0;
    for (std::size_t q = 0; q < nq; ++q) {
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                if (approx[q * k + i] == exact[q * k + j]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

struct Results {
    std::vector<RowId> ids;
    std::vector<float> distances;
};

Results search(const HnswIndex& index, const std::vector<float>& query, std::size_t k,
               std::size_t ef = 50) {
    Results r{std::vector<RowId>(k), std::vector<float>(k)};
    index.search(query, k, ef, r.ids, r.distances);
    return r;
}

}  // namespace

TEST(HnswIndex, RejectsBadSettings) {
    EXPECT_THROW(HnswIndex(0, Metric::L2, 10), std::invalid_argument);
    EXPECT_THROW(HnswIndex(4, Metric::L2, 0), std::invalid_argument);
    EXPECT_THROW(HnswIndex(4, Metric::L2, 10, HnswParams{1, 100, 1}), std::invalid_argument);
}

TEST(HnswIndex, EmptyIndexReturnsNoResults) {
    HnswIndex index(3, Metric::L2, 10);
    const auto r = search(index, {1, 2, 3}, 2);
    EXPECT_EQ(r.ids, (std::vector<RowId>{strata::kNoRow, strata::kNoRow}));
    EXPECT_EQ(r.distances[0], std::numeric_limits<float>::infinity());
}

TEST(HnswIndex, SingleVector) {
    HnswIndex index(2, Metric::L2, 10);
    index.add(std::vector<float>{3, 4});
    const auto r = search(index, {0, 0}, 3);
    EXPECT_EQ(r.ids, (std::vector<RowId>{0, strata::kNoRow, strata::kNoRow}));
    EXPECT_FLOAT_EQ(r.distances[0], 25.0f);
}

TEST(HnswIndex, CapacityIsEnforced) {
    HnswIndex index(2, Metric::L2, 3);
    index.add(std::vector<float>{1, 1, 2, 2});
    EXPECT_THROW(index.add(std::vector<float>{3, 3, 4, 4}), std::length_error);
    EXPECT_EQ(index.size(), 2u);
    index.add(std::vector<float>{3, 3});
    EXPECT_EQ(index.size(), 3u);
}

TEST(HnswIndex, HighRecallForEveryMetric) {
    constexpr std::size_t kDim = 24;
    const auto base = random_vectors(3000, kDim, 1);
    const auto queries = random_vectors(100, kDim, 2);
    for (auto metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        HnswIndex index(kDim, metric, 3000, HnswParams{16, 100, 7});
        index.add(base);
        EXPECT_GE(recall(index, base, queries, 10, 100), 0.95) << strata::metric_name(metric);
    }
}

TEST(HnswIndex, FindsEveryStoredVectorItself) {
    constexpr std::size_t kDim = 16;
    const auto base = random_vectors(2000, kDim, 3);
    HnswIndex index(kDim, Metric::L2, 2000);
    index.add(base);
    std::size_t found = 0;
    for (std::size_t i = 0; i < 2000; ++i) {
        const auto r = search(index, row(base, i, kDim), 1);
        found += (r.ids[0] == static_cast<RowId>(i));
    }
    EXPECT_GE(found, 1990u);
}

TEST(HnswIndex, ManyDuplicatesStillWork) {
    // Half the vectors are identical; the graph must stay searchable.
    std::vector<float> base;
    for (int i = 0; i < 500; ++i) {
        base.insert(base.end(), {1.0f, 1.0f, 1.0f, 1.0f});
    }
    const auto rest = random_vectors(500, 4, 4);
    base.insert(base.end(), rest.begin(), rest.end());

    HnswIndex index(4, Metric::L2, 1000);
    index.add(base);
    const auto r = search(index, {1, 1, 1, 1}, 10);
    for (float d : r.distances) {
        EXPECT_EQ(d, 0.0f);
    }
    EXPECT_GE(recall(index, base, random_vectors(50, 4, 5), 10, 200), 0.9);
}

TEST(HnswIndex, SameSeedGivesSameResults) {
    constexpr std::size_t kDim = 12;
    const auto base = random_vectors(1500, kDim, 6);
    const auto queries = random_vectors(30, kDim, 7);
    HnswIndex a(kDim, Metric::L2, 1500);
    HnswIndex b(kDim, Metric::L2, 1500);
    a.add(base);
    b.add(base);

    std::vector<RowId> ia(300), ib(300);
    std::vector<float> da(300), db(300);
    a.search_batch(queries, 10, 40, ia, da);
    b.search_batch(queries, 10, 40, ib, db);
    EXPECT_EQ(ia, ib);
    EXPECT_EQ(da, db);
}

TEST(HnswIndex, LevelsFollowExpectedSpread) {
    // About 1 in M nodes should reach level 1, and no node may exceed its link limit.
    constexpr std::size_t kN = 20000;
    HnswIndex index(4, Metric::L2, kN, HnswParams{16, 40, 9});
    index.add(random_vectors(kN, 4, 8), 0);
    const auto s = index.stats();
    ASSERT_GE(s.nodes_per_level.size(), 2u);
    const double share = static_cast<double>(s.nodes_per_level[1]) / kN;
    EXPECT_GT(share, 1.0 / 16 * 0.8);
    EXPECT_LT(share, 1.0 / 16 * 1.2);
    EXPECT_LE(s.max_links_level0, 32u);
    EXPECT_GT(s.avg_links_level0, 4.0);
}

TEST(HnswIndex, ParallelBuildMatchesSingleThreadRecall) {
    constexpr std::size_t kDim = 20;
    const auto base = random_vectors(5000, kDim, 10);
    const auto queries = random_vectors(100, kDim, 11);

    HnswIndex single(kDim, Metric::L2, 5000);
    single.add(base, 1);
    HnswIndex parallel(kDim, Metric::L2, 5000);
    parallel.add(base, 8);

    // Ids must still match input positions when built in parallel.
    const auto r = search(parallel, row(base, 1234, kDim), 1);
    EXPECT_EQ(r.ids[0], 1234);

    const double r1 = recall(single, base, queries, 10, 64);
    const double r8 = recall(parallel, base, queries, 10, 64);
    EXPECT_NEAR(r1, r8, 0.02);
}

TEST(HnswIndex, SearchWhileAdding) {
    constexpr std::size_t kDim = 8;
    constexpr std::size_t kN = 4000;
    const auto base = random_vectors(kN, kDim, 12);
    HnswIndex index(kDim, Metric::L2, kN);

    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (std::size_t i = 0; i < kN; i += 100) {
            index.add(std::span(base).subspan(i * kDim, 100 * kDim), 2);
        }
        done = true;
    });

    // Searches running during inserts must only return valid ids.
    std::vector<std::jthread> readers;
    readers.reserve(2);
    for (int t = 0; t < 2; ++t) {
        readers.emplace_back([&, t] {
            const auto queries = random_vectors(50, kDim, 100 + static_cast<unsigned>(t));
            std::size_t q = 0;
            while (!done) {
                const auto r = search(index, row(queries, q++ % 50, kDim), 5, 20);
                for (RowId id : r.ids) {
                    ASSERT_TRUE(id == strata::kNoRow ||
                                (id >= 0 && static_cast<std::size_t>(id) < index.size()));
                }
            }
        });
    }
    writer.join();
    readers.clear();
    EXPECT_GE(recall(index, base, random_vectors(50, kDim, 13), 10, 64), 0.9);
}

TEST(HnswIndex, SaveAndLoadGiveIdenticalResults) {
    constexpr std::size_t kDim = 10;
    const auto base = random_vectors(1000, kDim, 14);
    const auto queries = random_vectors(20, kDim, 15);
    HnswIndex index(kDim, Metric::Cosine, 1000);
    index.add(base);

    const auto path = (std::filesystem::temp_directory_path() / "strata_hnsw_test.bin").string();
    index.save(path);
    const auto loaded = HnswIndex::load(path, 2000);
    EXPECT_EQ(loaded->size(), 1000u);
    EXPECT_EQ(loaded->capacity(), 2000u);
    EXPECT_EQ(loaded->metric(), Metric::Cosine);

    std::vector<RowId> a(200), b(200);
    std::vector<float> da(200), db(200);
    index.search_batch(queries, 10, 50, a, da);
    loaded->search_batch(queries, 10, 50, b, db);
    EXPECT_EQ(a, b);
    EXPECT_EQ(da, db);

    // A loaded index can keep growing.
    loaded->add(random_vectors(10, kDim, 16));
    EXPECT_EQ(loaded->size(), 1010u);
    std::filesystem::remove(path);
}

TEST(HnswIndex, LoadRejectsDamagedFiles) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto junk = (dir / "strata_hnsw_junk.bin").string();
    std::ofstream(junk) << "definitely not an index";
    EXPECT_THROW(HnswIndex::load(junk), std::runtime_error);

    // A valid file cut in half must also be rejected.
    HnswIndex index(4, Metric::L2, 100);
    index.add(random_vectors(100, 4, 17));
    const auto full = (dir / "strata_hnsw_full.bin").string();
    index.save(full);
    std::filesystem::resize_file(full, std::filesystem::file_size(full) / 2);
    EXPECT_THROW(HnswIndex::load(full), std::runtime_error);

    std::filesystem::remove(junk);
    std::filesystem::remove(full);
}
