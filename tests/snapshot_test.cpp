#include "strata/snapshot.h"

#include <gtest/gtest.h>

#include <atomic>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "strata/file_system.h"
#include "strata/hnsw_index.h"

using strata::HnswIndex;
using strata::MemoryFileSystem;
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

std::vector<RowId> top_ids(const HnswIndex& index, const std::vector<float>& queries) {
    const std::size_t nq = queries.size() / index.dim();
    std::vector<RowId> ids(nq * 5);
    std::vector<float> dist(nq * 5);
    index.search_batch(queries, 5, 40, ids, dist);
    return ids;
}

}  // namespace

TEST(SnapshotHeader, RoundTrip) {
    strata::SnapshotHeader h;
    h.metric = Metric::Cosine;
    h.encoding = strata::Encoding::SQ8;
    h.dim = 8;
    h.M = 4;
    h.ef_construction = 20;
    h.seed = 9;
    h.capacity = 10;
    h.count = 2;
    h.entry = 1;
    h.wal_lsn = 77;
    h.code_size = 8;
    const std::uint64_t sizes[] = {64, 16, 2, 72, 0};  // 72 = 2 nodes * (1 + 2M) links * 4 bytes
    std::uint64_t offset = strata::kSnapshotHeaderBytes;
    for (std::size_t i = 0; i < strata::kSectionCount; ++i) {
        h.sections[i] = {offset, sizes[i], static_cast<std::uint32_t>(i * 11)};
        offset += sizes[i];
    }
    const auto bytes = strata::encode_snapshot_header(h);
    const auto back = strata::parse_snapshot_header(bytes);
    EXPECT_EQ(back.metric, h.metric);
    EXPECT_EQ(back.encoding, h.encoding);
    EXPECT_EQ(back.wal_lsn, 77u);
    EXPECT_EQ(back.sections[strata::kLayer0LinksSection].size, sizes[3]);
    EXPECT_EQ(back.sections[2].crc, 22u);

    // Any changed byte breaks the header checksum (or the magic/version check).
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        auto bad = bytes;
        bad[i] ^= 0x01;
        EXPECT_THROW(strata::parse_snapshot_header(bad), std::runtime_error) << "byte " << i;
    }
}

TEST(Snapshot, EveryDamagedByteIsCaught) {
    constexpr std::size_t kDim = 4;
    HnswIndex index(kDim, Metric::L2, 60, {4, 20, 1});
    index.add(random_vectors(60, kDim, 1));
    MemoryFileSystem fs;
    fs.make_dir("/d");
    index.save_snapshot(fs, "/d/s.snap", index.save_point(), 5);
    const auto clean = fs.raw_contents("/d/s.snap");

    for (std::size_t i = 0; i < clean.size(); ++i) {
        fs.raw_contents("/d/s.snap") = clean;
        fs.raw_contents("/d/s.snap")[i] ^= 0x40;
        EXPECT_THROW(HnswIndex::load_snapshot(fs, "/d/s.snap"), std::runtime_error) << "byte " << i;
    }
    fs.raw_contents("/d/s.snap") = clean;
    const auto loaded = HnswIndex::load_snapshot(fs, "/d/s.snap");
    EXPECT_EQ(loaded.wal_lsn, 5u);
    EXPECT_EQ(loaded.index->size(), 60u);
}

TEST(Snapshot, CutOffOrExtendedFileIsRejected) {
    HnswIndex index(4, Metric::L2, 30);
    index.add(random_vectors(30, 4, 2));
    MemoryFileSystem fs;
    fs.make_dir("/d");
    index.save_snapshot(fs, "/d/s.snap", index.save_point(), 0);
    const auto clean = fs.raw_contents("/d/s.snap");
    for (std::size_t len : {std::size_t{0}, std::size_t{100}, clean.size() - 1}) {
        fs.raw_contents("/d/s.snap").assign(clean.begin(), clean.begin() + static_cast<long>(len));
        EXPECT_THROW(HnswIndex::load_snapshot(fs, "/d/s.snap"), std::runtime_error);
    }
    fs.raw_contents("/d/s.snap") = clean;
    fs.raw_contents("/d/s.snap").push_back(0);
    EXPECT_THROW(HnswIndex::load_snapshot(fs, "/d/s.snap"), std::runtime_error);
}

TEST(Snapshot, EmptyIndex) {
    HnswIndex index(3, Metric::InnerProduct, 10);
    MemoryFileSystem fs;
    fs.make_dir("/d");
    index.save_snapshot(fs, "/d/s.snap", index.save_point(), 0);
    const auto loaded = HnswIndex::load_snapshot(fs, "/d/s.snap", 50);
    EXPECT_EQ(loaded.index->size(), 0u);
    EXPECT_EQ(loaded.index->capacity(), 50u);
    loaded.index->add(random_vectors(5, 3, 3));
    EXPECT_EQ(loaded.index->size(), 5u);
}

TEST(Snapshot, SavingWhileAddingGivesAValidIndex) {
    constexpr std::size_t kDim = 8;
    const auto base = random_vectors(6000, kDim, 6);
    const auto queries = random_vectors(30, kDim, 7);
    HnswIndex index(kDim, Metric::L2, 6000);
    index.add(std::span(base).first(2000 * kDim), 4);
    const auto point = index.save_point();

    // Keep adding while the snapshot is written; links to the new nodes are left out.
    std::jthread adder([&] { index.add(std::span(base).subspan(2000 * kDim), 4); });
    MemoryFileSystem fs;
    fs.make_dir("/d");
    index.save_snapshot(fs, "/d/s.snap", point, 0);
    adder.join();

    const auto loaded = HnswIndex::load_snapshot(fs, "/d/s.snap");
    EXPECT_EQ(loaded.index->size(), 2000u);
    for (RowId id : top_ids(*loaded.index, queries)) {
        EXPECT_LT(id, 2000);
    }
    // The loaded index finds a stored vector exactly.
    std::vector<RowId> ids(1);
    std::vector<float> dist(1);
    loaded.index->search(std::span(base).subspan(1234 * kDim, kDim), 1, 40, ids, dist);
    EXPECT_EQ(ids[0], 1234);
}
