#include "strata/hnsw_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "strata/distance.h"
#include "strata/flat_index.h"
#include "strata/quantizer.h"

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

namespace {

std::vector<float> normalized_copy(std::vector<float> data, std::size_t dim) {
    for (std::size_t i = 0; i < data.size() / dim; ++i) {
        strata::normalize(data.data() + i * dim, dim);
    }
    return data;
}

// Recall of search_rerank against exact results.
double rerank_recall(const HnswIndex& index, const std::vector<float>& base,
                     const std::vector<float>& queries, std::size_t k, std::size_t ef,
                     std::size_t rerank) {
    const std::size_t dim = index.dim();
    FlatIndex flat(dim, index.metric());
    flat.add(base);
    const strata::FullVectors full{base.data(), dim};
    std::size_t hits = 0;
    const std::size_t nq = queries.size() / dim;
    for (std::size_t q = 0; q < nq; ++q) {
        const auto query = row(queries, q, dim);
        std::vector<RowId> exact(k), approx(k);
        std::vector<float> d1(k), d2(k);
        flat.search(query, k, exact, d1);
        index.search_rerank(query, k, ef, rerank, full, approx, d2);
        for (RowId id : approx) {
            hits += std::find(exact.begin(), exact.end(), id) != exact.end();
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

}  // namespace

TEST(HnswIndex, Sq8StoresOneBytePerValue) {
    constexpr std::size_t kDim = 24;
    const auto base = random_vectors(3000, kDim, 20);
    const auto queries = random_vectors(100, kDim, 21);
    HnswIndex index(kDim, Metric::L2, 3000, HnswParams{},
                    strata::ScalarQuantizer::train(base, kDim));
    index.add(base, 4);
    EXPECT_EQ(index.encoding(), strata::Encoding::SQ8);
    EXPECT_EQ(index.bytes_per_vector(), kDim);
    EXPECT_GE(recall(index, base, queries, 10, 100), 0.85);
    EXPECT_GE(rerank_recall(index, base, queries, 10, 100, 40), 0.95);
}

TEST(HnswIndex, PqWithRerankingRecoversAccuracy) {
    constexpr std::size_t kDim = 32;
    const auto base = random_vectors(4000, kDim, 22);
    const auto queries = random_vectors(100, kDim, 23);
    HnswIndex index(kDim, Metric::L2, 4000, HnswParams{},
                    strata::ProductQuantizer::train(base, kDim, {8, 15, 1, 0}));
    index.add(base, 4);
    EXPECT_EQ(index.bytes_per_vector(), 8u);  // 128 bytes of floats -> 8 bytes

    const double plain = recall(index, base, queries, 10, 100);
    const double reranked = rerank_recall(index, base, queries, 10, 200, 100);
    EXPECT_GT(reranked, plain + 0.1);
    EXPECT_GE(reranked, 0.9);
}

TEST(HnswIndex, CompressedCosineWorks) {
    constexpr std::size_t kDim = 32;
    const auto base = random_vectors(3000, kDim, 24);
    const auto queries = random_vectors(50, kDim, 25);
    // Quantizers are trained on normalized vectors, since that's what a Cosine index stores.
    const auto sample = normalized_copy(base, kDim);
    HnswIndex sq8(kDim, Metric::Cosine, 3000, HnswParams{},
                  strata::ScalarQuantizer::train(sample, kDim));
    HnswIndex pq(kDim, Metric::Cosine, 3000, HnswParams{},
                 strata::ProductQuantizer::train(sample, kDim, {16, 15, 1, 0}));
    sq8.add(base, 4);
    pq.add(base, 4);
    EXPECT_GE(rerank_recall(sq8, base, queries, 10, 100, 40), 0.95);
    EXPECT_GE(rerank_recall(pq, base, queries, 10, 200, 100), 0.9);
}

TEST(HnswIndex, RerankGivesExactDistances) {
    constexpr std::size_t kDim = 16;
    const auto base = random_vectors(500, kDim, 26);
    HnswIndex index(kDim, Metric::InnerProduct, 500, HnswParams{},
                    strata::ScalarQuantizer::train(base, kDim));
    index.add(base);
    const auto query = row(random_vectors(1, kDim, 27), 0, kDim);
    std::vector<RowId> ids(5);
    std::vector<float> dist(5);
    index.search_rerank(query, 5, 50, 20, {base.data(), kDim}, ids, dist);
    for (std::size_t i = 0; i < 5; ++i) {
        const auto v = row(base, static_cast<std::size_t>(ids[i]), kDim);
        EXPECT_FLOAT_EQ(dist[i], -strata::dot_product(query.data(), v.data(), kDim));
    }
    EXPECT_TRUE(std::is_sorted(dist.begin(), dist.end()));
}

TEST(HnswIndex, CompressedIndexRejectsBadUse) {
    constexpr std::size_t kDim = 8;
    const auto base = random_vectors(300, kDim, 28);
    const auto sq = strata::ScalarQuantizer::train(base, kDim);
    EXPECT_THROW(HnswIndex(16, Metric::L2, 10, HnswParams{}, sq), std::invalid_argument);

    HnswIndex index(kDim, Metric::L2, 300, HnswParams{}, sq);
    index.add(base);

    std::vector<RowId> ids(3);
    std::vector<float> dist(3);
    EXPECT_THROW(index.search_rerank(row(base, 0, kDim), 3, 10, 10, {}, ids, dist),
                 std::invalid_argument);
}

TEST(HnswIndex, CompressedIndexesSaveAndLoad) {
    constexpr std::size_t kDim = 16;
    const auto base = random_vectors(1000, kDim, 29);
    const auto queries = random_vectors(20, kDim, 30);
    std::vector<std::unique_ptr<HnswIndex>> indexes;
    indexes.push_back(std::make_unique<HnswIndex>(kDim, Metric::L2, 1000, HnswParams{},
                                                  strata::ScalarQuantizer::train(base, kDim)));
    indexes.push_back(
        std::make_unique<HnswIndex>(kDim, Metric::InnerProduct, 1000, HnswParams{},
                                    strata::ProductQuantizer::train(base, kDim, {4, 10, 1, 0})));

    const auto path = (std::filesystem::temp_directory_path() / "strata_compressed.snap").string();
    for (const auto& index : indexes) {
        index->add(base, 4);
        index->save(path);
        const auto loaded = HnswIndex::load(path);
        EXPECT_EQ(loaded->encoding(), index->encoding());
        EXPECT_EQ(loaded->capacity(), 1000u);

        std::vector<RowId> a(200), b(200);
        std::vector<float> da(200), db(200);
        index->search_batch(queries, 10, 50, a, da);
        loaded->search_batch(queries, 10, 50, b, db);
        EXPECT_EQ(a, b);
        EXPECT_EQ(da, db);

        std::vector<float> v1(kDim), v2(kDim);
        index->get_vector(123, v1);
        loaded->get_vector(123, v2);
        EXPECT_EQ(v1, v2);
    }
    std::filesystem::remove(path);
}

TEST(HnswIndex, GetVectorReturnsStoredValues) {
    constexpr std::size_t kDim = 6;
    const auto base = random_vectors(50, kDim, 31);
    HnswIndex l2(kDim, Metric::L2, 50);
    HnswIndex cosine(kDim, Metric::Cosine, 50);
    l2.add(base);
    cosine.add(base);
    std::vector<float> out(kDim);
    l2.get_vector(7, out);
    EXPECT_EQ(out, row(base, 7, kDim));
    cosine.get_vector(7, out);
    auto expected = row(base, 7, kDim);
    strata::normalize(expected.data(), kDim);
    for (std::size_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(out[i], expected[i]);
    }
    EXPECT_THROW(l2.get_vector(50, out), std::invalid_argument);
}

TEST(HnswIndex, ReserveThenInsert) {
    constexpr std::size_t kDim = 8;
    const auto base = random_vectors(200, kDim, 32);
    HnswIndex index(kDim, Metric::L2, 200);
    const RowId first = index.reserve(100);
    const RowId second = index.reserve(100);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(second, 100);
    // Filling the second block first must still give every row its own vector.
    index.insert_reserved(second, std::span(base).subspan(100 * kDim));
    index.insert_reserved(first, std::span(base).subspan(0, 100 * kDim));
    const auto r = search(index, row(base, 150, kDim), 1);
    EXPECT_EQ(r.ids[0], 150);
    EXPECT_THROW(index.reserve(1), std::length_error);
    EXPECT_THROW(index.insert_reserved(150, std::span(base).subspan(0, 100 * kDim)),
                 std::invalid_argument);
}
