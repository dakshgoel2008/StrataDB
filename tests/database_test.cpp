#include "strata/database.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "strata/quantizer.h"

using strata::Database;
using strata::DatabaseOptions;
using strata::HnswIndex;
using strata::HnswParams;
using strata::MemoryFileSystem;
using strata::Metric;
using strata::RowId;
using strata::SyncMode;

namespace {

constexpr std::size_t kDim = 8;

// The vector stored at a row in these tests. Any row can be checked on its own.
std::vector<float> row_vector(std::uint64_t row) {
    std::vector<float> v(kDim);
    std::mt19937_64 rng(row * 7919 + 1);
    for (float& x : v) {
        x = static_cast<float>(rng() % 10000) / 100.0f;
    }
    return v;
}

std::vector<float> rows(std::uint64_t first, std::size_t n) {
    std::vector<float> out;
    for (std::size_t i = 0; i < n; ++i) {
        const auto v = row_vector(first + i);
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::unique_ptr<HnswIndex> empty_index(std::size_t capacity = 20000) {
    return std::make_unique<HnswIndex>(kDim, Metric::L2, capacity, HnswParams{8, 40, 1});
}

// Every row holds the vector row_vector() gives for it.
void expect_rows_intact(const Database& db) {
    std::vector<float> got(kDim);
    for (std::size_t r = 0; r < db.size(); ++r) {
        db.index().get_vector(static_cast<RowId>(r), got);
        ASSERT_EQ(got, row_vector(r)) << "row " << r;
    }
}

class TempDir {
public:
    TempDir() : path_((std::filesystem::temp_directory_path() / "strata_db_test").string()) {
        std::filesystem::remove_all(path_);
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::vector<std::string> sorted_names(strata::FileSystem& fs, const std::string& dir) {
    auto names = fs.list(dir);
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace

TEST(Database, InsertCloseReopen) {
    TempDir dir;
    {
        auto db = Database::create(dir.path(), empty_index());
        for (std::uint64_t r = 0; r < 300; r += 30) {
            EXPECT_EQ(db->insert(rows(r, 30)), static_cast<RowId>(r));
        }
    }
    auto db = Database::open(dir.path());
    EXPECT_EQ(db->size(), 300u);
    EXPECT_EQ(db->recovery().replayed_records, 10u);
    EXPECT_EQ(db->recovery().replayed_rows, 300u);
    expect_rows_intact(*db);

    std::vector<RowId> ids(1);
    std::vector<float> dist(1);
    db->search(row_vector(123), 1, 40, ids, dist);
    EXPECT_EQ(ids[0], 123);
    EXPECT_EQ(db->insert(rows(300, 1)), 300);
}

TEST(Database, CheckpointKeepsExactGraphAndCleansUp) {
    TempDir dir;
    auto& fs = strata::local_file_system();
    std::vector<RowId> before(50);
    std::vector<float> d1(50);
    const auto queries = rows(5000, 10);
    {
        auto db = Database::create(dir.path(), empty_index(), {SyncMode::Group, 4096});
        for (std::uint64_t r = 0; r < 500; r += 50) {
            db->insert(rows(r, 50));
        }
        EXPECT_GT(fs.list(dir.path()).size(), 4u);  // several log files
        db->checkpoint();
        EXPECT_EQ(sorted_names(fs, dir.path()),
                  (std::vector<std::string>{"LOCK", "MANIFEST", strata::snapshot_file_name(2),
                                            strata::wal_file_name(11)}));
        db->index().search_batch(queries, 5, 40, before, d1);
    }
    auto db = Database::open(dir.path());
    EXPECT_EQ(db->recovery().snapshot_rows, 500u);
    EXPECT_EQ(db->recovery().replayed_records, 0u);
    std::vector<RowId> after(50);
    std::vector<float> d2(50);
    db->index().search_batch(queries, 5, 40, after, d2);
    EXPECT_EQ(before, after);  // loaded from the snapshot, so the graph is the same
}

TEST(Database, CutOffRecordIsDroppedOnOpen) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    {
        auto db = Database::create("/db", empty_index(), options);
        db->insert(rows(0, 10));
        db->insert(rows(10, 10));
    }
    // Half of a third record, as if the program died while writing it.
    auto& log = fs.raw_contents("/db/" + strata::wal_file_name(1));
    const auto extra = rows(20, 10);
    std::vector<std::uint8_t> record;
    strata::encode_wal_record(3, strata::WalRecordType::Insert,
                              strata::encode_insert(20, extra, kDim), record);
    log.insert(log.end(), record.begin(), record.begin() + 40);

    {
        auto db = Database::open("/db", options);
        EXPECT_EQ(db->size(), 20u);
        EXPECT_EQ(db->recovery().dropped_bytes, 40u);
        db->insert(rows(20, 5));
    }
    auto db = Database::open("/db", options);
    EXPECT_EQ(db->size(), 25u);
    EXPECT_EQ(db->recovery().dropped_bytes, 0u);
    expect_rows_intact(*db);
}

TEST(Database, LargeInsertIsSplitIntoRecords) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    constexpr std::size_t kBigDim = 1024;  // 4 KiB per row, so about 1000 rows per record
    {
        auto db = Database::create(
            "/db", std::make_unique<HnswIndex>(kBigDim, Metric::L2, 3000, HnswParams{8, 20, 1}),
            options);
        std::vector<float> data(2100 * kBigDim);
        std::mt19937 rng(1);
        for (float& x : data) {
            x = static_cast<float>(rng() % 1000);
        }
        EXPECT_EQ(db->insert(data), 0);
        EXPECT_EQ(db->wal().record_count(), 3u);
    }
    auto db = Database::open("/db", options);
    EXPECT_EQ(db->size(), 2100u);
    EXPECT_EQ(db->recovery().replayed_records, 3u);
}

TEST(Database, DiskErrorStopsWritesUntilReopened) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    {
        auto db = Database::create("/db", empty_index(), options);
        db->insert(rows(0, 5));
        fs.fail_next_sync();
        EXPECT_THROW(db->insert(rows(5, 5)), std::runtime_error);
        EXPECT_THROW(db->insert(rows(5, 5)), std::runtime_error);
        EXPECT_THROW(db->checkpoint(), std::runtime_error);
        // Searches still work.
        std::vector<RowId> ids(1);
        std::vector<float> dist(1);
        db->search(row_vector(3), 1, 20, ids, dist);
        EXPECT_EQ(ids[0], 3);
    }
    auto db = Database::open("/db", options);
    EXPECT_GE(db->size(), 5u);  // the failed insert may or may not have reached the disk
    expect_rows_intact(*db);
    db->insert(rows(db->size(), 1));
}

TEST(Database, RefusesBadDirectories) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    EXPECT_THROW(Database::open("/nothing", options), std::runtime_error);
    Database::create("/db", empty_index(), options);
    EXPECT_THROW(Database::create("/db", empty_index(), options), std::runtime_error);
    EXPECT_THROW(Database::create("/other", nullptr, options), std::invalid_argument);
}

TEST(Database, MissingLogFileIsReported) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    options.wal_file_bytes = 200;
    {
        auto db = Database::create("/db", empty_index(), options);
        for (std::uint64_t r = 0; r < 20; ++r) {
            db->insert(rows(r, 1));
        }
    }
    auto names = sorted_names(fs, "/db");
    std::vector<std::string> logs;
    std::copy_if(names.begin(), names.end(), std::back_inserter(logs),
                 [](const std::string& n) { return n.starts_with("wal_"); });
    ASSERT_GT(logs.size(), 3u);
    fs.remove("/db/" + logs[1]);
    EXPECT_THROW(Database::open("/db", options), std::runtime_error);
}

TEST(Database, DamagedRecordInOlderLogFileIsReported) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    options.wal_file_bytes = 200;
    {
        auto db = Database::create("/db", empty_index(), options);
        for (std::uint64_t r = 0; r < 20; ++r) {
            db->insert(rows(r, 1));
        }
    }
    fs.raw_contents("/db/" + strata::wal_file_name(1))[20] ^= 1;
    EXPECT_THROW(Database::open("/db", options), std::runtime_error);
}

TEST(Database, StartsFromAPrebuiltIndexAndGrowsCapacity) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    auto index = empty_index(100);
    index->add(rows(0, 100));
    {
        auto db = Database::create("/db", std::move(index), options);
        EXPECT_THROW(db->insert(rows(100, 1)), std::length_error);  // full
    }
    options.capacity = 200;
    auto db = Database::open("/db", options);
    EXPECT_EQ(db->index().capacity(), 200u);
    db->insert(rows(100, 100));
    expect_rows_intact(*db);
}

TEST(Database, CompressedIndexSurvivesRecovery) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    const auto sample = rows(0, 1000);
    std::vector<float> before(kDim);
    {
        auto db = Database::create(
            "/db",
            std::make_unique<HnswIndex>(kDim, Metric::L2, 2000, HnswParams{8, 40, 1},
                                        strata::ScalarQuantizer::train(sample, kDim)),
            options);
        db->insert(std::span(sample).first(500 * kDim));
        db->checkpoint();
        db->insert(std::span(sample).subspan(500 * kDim));
        db->index().get_vector(777, before);
    }
    auto db = Database::open("/db", options);
    EXPECT_EQ(db->index().encoding(), strata::Encoding::SQ8);
    EXPECT_EQ(db->size(), 1000u);
    std::vector<float> after(kDim);
    db->index().get_vector(777, after);
    EXPECT_EQ(before, after);  // the same quantizer came back from the snapshot
}

TEST(Database, ConcurrentInsertsAndCheckpoints) {
    MemoryFileSystem fs;
    DatabaseOptions options;
    options.fs = &fs;
    options.wal_file_bytes = 4096;
    constexpr int kThreads = 6;
    constexpr int kPerThread = 150;
    std::mutex mutex;
    std::map<RowId, std::vector<float>> acked;
    {
        auto db = Database::create("/db", empty_index(), options);
        std::vector<std::jthread> threads;
        threads.reserve(kThreads + 1);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                for (int i = 0; i < kPerThread; ++i) {
                    const auto v = row_vector(static_cast<std::uint64_t>(t) * 100000 +
                                              static_cast<std::uint64_t>(i));
                    const RowId row = db->insert(v);
                    std::lock_guard lock(mutex);
                    acked[row] = v;
                }
            });
        }
        threads.emplace_back([&] {
            for (int i = 0; i < 10; ++i) {
                db->checkpoint();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    auto db = Database::open("/db", options);
    ASSERT_EQ(db->size(), static_cast<std::size_t>(kThreads * kPerThread));
    std::vector<float> got(kDim);
    for (const auto& [row, v] : acked) {
        db->index().get_vector(row, got);
        ASSERT_EQ(got, v) << "row " << row;
    }
}

TEST(ManifestFormat, RoundTripAndDamage) {
    const auto bytes = strata::encode_manifest({7, 1234});
    const auto m = strata::parse_manifest(bytes);
    EXPECT_EQ(m.snapshot_id, 7u);
    EXPECT_EQ(m.wal_lsn, 1234u);
    for (std::size_t i = 0; i < 36; ++i) {
        auto bad = bytes;
        bad[i] ^= 0x08;
        EXPECT_THROW(strata::parse_manifest(bad), std::runtime_error) << "byte " << i;
    }
    EXPECT_THROW(strata::parse_manifest(std::span(bytes).first(20)), std::runtime_error);
}

TEST(InsertRecordFormat, RoundTripAndBadSizes) {
    const auto data = rows(0, 3);
    const auto payload = strata::encode_insert(42, data, kDim);
    strata::InsertRecord rec;
    ASSERT_TRUE(strata::decode_insert(payload, rec));
    EXPECT_EQ(rec.first_row, 42u);
    EXPECT_EQ(rec.count, 3u);
    EXPECT_EQ(rec.dim, kDim);
    EXPECT_FALSE(strata::decode_insert(std::span(payload).first(payload.size() - 1), rec));
    EXPECT_FALSE(strata::decode_insert(std::span(payload).first(10), rec));
}
