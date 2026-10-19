#include "strata/wal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using strata::MemoryFileSystem;
using strata::SyncMode;
using strata::WalRecord;
using strata::WalRecordType;
using strata::WriteAheadLog;

namespace {

std::vector<std::uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

std::vector<std::uint8_t> encode_many(std::uint64_t first, std::size_t n) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i < n; ++i) {
        strata::encode_wal_record(first + i, WalRecordType::Insert,
                                  bytes("record " + std::to_string(first + i)), out);
    }
    return out;
}

std::vector<std::string> read_all(std::span<const std::uint8_t> data, std::uint64_t first,
                                  std::size_t* good = nullptr) {
    std::vector<std::string> out;
    const std::size_t n = strata::read_wal_records(data, first, [&](const WalRecord& r) {
        out.emplace_back(r.payload.begin(), r.payload.end());
    });
    if (good != nullptr) {
        *good = n;
    }
    return out;
}

// Every record from all log files in the directory, in order.
std::vector<std::string> read_dir(MemoryFileSystem& fs, const std::string& dir) {
    std::vector<std::uint64_t> firsts;
    for (const auto& name : fs.list(dir)) {
        std::uint64_t first = 0;
        if (strata::parse_wal_file_name(name, first)) {
            firsts.push_back(first);
        }
    }
    std::sort(firsts.begin(), firsts.end());
    std::vector<std::string> out;
    for (auto first : firsts) {
        const auto data = strata::read_whole_file(fs, dir + "/" + strata::wal_file_name(first));
        for (auto& s : read_all(data, first)) {
            out.push_back(std::move(s));
        }
    }
    return out;
}

}  // namespace

TEST(WalFormat, RoundTrip) {
    const auto data = encode_many(5, 3);
    std::size_t good = 0;
    const auto records = read_all(data, 5, &good);
    EXPECT_EQ(records, (std::vector<std::string>{"record 5", "record 6", "record 7"}));
    EXPECT_EQ(good, data.size());
}

TEST(WalFormat, StopsAtACutOffRecord) {
    const auto data = encode_many(1, 3);
    const std::size_t two = encode_many(1, 2).size();
    for (std::size_t cut = two; cut < data.size(); ++cut) {
        std::size_t good = 0;
        const auto records = read_all(std::span(data).first(cut), 1, &good);
        EXPECT_EQ(records.size(), 2u);
        EXPECT_EQ(good, two);
    }
}

TEST(WalFormat, StopsAtAnyFlippedBit) {
    const auto clean = encode_many(1, 2);
    const std::size_t one = encode_many(1, 1).size();
    for (std::size_t i = one; i < clean.size(); ++i) {
        auto data = clean;
        data[i] ^= 0x10;
        std::size_t good = 0;
        EXPECT_EQ(read_all(data, 1, &good).size(), 1u) << "byte " << i;
        EXPECT_EQ(good, one);
    }
}

TEST(WalFormat, StopsAtUnexpectedLsn) {
    auto data = encode_many(1, 2);
    const auto later = encode_many(7, 1);  // skips 3..6
    data.insert(data.end(), later.begin(), later.end());
    EXPECT_EQ(read_all(data, 1).size(), 2u);
    EXPECT_TRUE(read_all(data, 2).empty());  // first record must have the given lsn
}

TEST(WalFormat, HugeLengthIsTreatedAsDamage) {
    auto data = encode_many(1, 1);
    data[3] = 0xff;  // length becomes larger than allowed
    EXPECT_TRUE(read_all(data, 1).empty());
}

TEST(WalFormat, FileNames) {
    EXPECT_EQ(strata::wal_file_name(42), "wal_00000000000000000042.log");
    std::uint64_t lsn = 0;
    EXPECT_TRUE(strata::parse_wal_file_name("wal_00000000000000000042.log", lsn));
    EXPECT_EQ(lsn, 42u);
    EXPECT_FALSE(strata::parse_wal_file_name("wal_42.log", lsn));
    EXPECT_FALSE(strata::parse_wal_file_name("wal_0000000000000000004x.log", lsn));
    EXPECT_FALSE(strata::parse_wal_file_name("wal_99999999999999999999.log", lsn));  // > 2^64
    EXPECT_FALSE(strata::parse_wal_file_name("snapshot_00000000000000000042.snap", lsn));
}

TEST(WalFormat, SyncModeNames) {
    for (auto m : {SyncMode::EveryWrite, SyncMode::Group, SyncMode::None}) {
        EXPECT_EQ(strata::parse_sync_mode(strata::sync_mode_name(m)), m);
    }
    EXPECT_THROW(strata::parse_sync_mode("sometimes"), std::invalid_argument);
}

TEST(WriteAheadLog, AppendWaitAndReadBack) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    {
        WriteAheadLog wal(fs, "/db", 1, {}, false);
        for (int i = 1; i <= 3; ++i) {
            const auto payload = bytes("record " + std::to_string(i));
            const std::uint64_t lsn = wal.append(WalRecordType::Insert, payload);
            EXPECT_EQ(lsn, static_cast<std::uint64_t>(i));
            wal.wait(lsn);
        }
        EXPECT_EQ(wal.last_lsn(), 3u);
        EXPECT_EQ(wal.sync_count(), 3u);
    }
    EXPECT_EQ(read_dir(fs, "/db"), (std::vector<std::string>{"record 1", "record 2", "record 3"}));
}

TEST(WriteAheadLog, AppendToLastFileContinuesIt) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    {
        WriteAheadLog wal(fs, "/db", 1, {}, false);
        wal.wait(wal.append(WalRecordType::Insert, bytes("record 1")));
    }
    {
        WriteAheadLog wal(fs, "/db", 2, {}, true);
        wal.wait(wal.append(WalRecordType::Insert, bytes("record 2")));
    }
    EXPECT_EQ(fs.list("/db").size(), 1u);
    EXPECT_EQ(read_dir(fs, "/db"), (std::vector<std::string>{"record 1", "record 2"}));
}

TEST(WriteAheadLog, RollsOverAndRemovesOldFiles) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::Group, 100}, false);
    for (int i = 1; i <= 20; ++i) {
        wal.wait(wal.append(WalRecordType::Insert, bytes("record " + std::to_string(i))));
    }
    const std::size_t files = fs.list("/db").size();
    EXPECT_GT(files, 3u);
    EXPECT_EQ(read_dir(fs, "/db").size(), 20u);

    wal.start_new_file();
    wal.remove_files_through(20);
    EXPECT_EQ(fs.list("/db"), std::vector<std::string>{strata::wal_file_name(21)});
    wal.wait(wal.append(WalRecordType::Insert, bytes("record 21")));
    EXPECT_EQ(read_dir(fs, "/db"), std::vector<std::string>{"record 21"});
}

TEST(WriteAheadLog, RemoveKeepsFilesStillNeeded) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {}, false);
    for (int i = 1; i <= 4; ++i) {
        wal.wait(wal.append(WalRecordType::Insert, bytes("r")));
        wal.start_new_file();  // files start at 1, 2, 3, 4, 5
    }
    wal.remove_files_through(2);  // records 3 and 4 are still needed
    EXPECT_EQ(read_dir(fs, "/db").size(), 2u);
    EXPECT_EQ(fs.list("/db").size(), 3u);
}

TEST(WriteAheadLog, GroupCommitSharesSyncs) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::Group}, false);
    constexpr int kThreads = 8;
    constexpr int kPerThread = 200;
    {
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&wal, t] {
                for (int i = 0; i < kPerThread; ++i) {
                    const auto payload = bytes(std::to_string(t) + ":" + std::to_string(i));
                    wal.wait(wal.append(WalRecordType::Insert, payload));
                }
            });
        }
    }
    const auto records = read_dir(fs, "/db");
    ASSERT_EQ(records.size(), static_cast<std::size_t>(kThreads * kPerThread));
    // Each thread's records appear in the order it wrote them.
    for (int t = 0; t < kThreads; ++t) {
        int next = 0;
        for (const auto& r : records) {
            if (r.starts_with(std::to_string(t) + ":")) {
                EXPECT_EQ(r, std::to_string(t) + ":" + std::to_string(next++));
            }
        }
        EXPECT_EQ(next, kPerThread);
    }
    EXPECT_LE(wal.sync_count(), wal.record_count());
}

TEST(WriteAheadLog, NoSyncModeNeverSyncs) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::None}, false);
    for (int i = 0; i < 10; ++i) {
        wal.wait(wal.append(WalRecordType::Insert, bytes("r")));
    }
    EXPECT_EQ(wal.sync_count(), 0u);
    EXPECT_EQ(read_dir(fs, "/db").size(), 10u);  // written, just not synced
}

TEST(WriteAheadLog, SyncFailureStopsTheLog) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::EveryWrite}, false);
    wal.append(WalRecordType::Insert, bytes("ok"));
    fs.fail_next_sync();
    EXPECT_THROW(wal.append(WalRecordType::Insert, bytes("lost")), std::runtime_error);
    // The disk works again, but after a failed sync nobody knows what reached it,
    // so the log refuses to go on.
    EXPECT_THROW(wal.append(WalRecordType::Insert, bytes("later")), std::runtime_error);
    EXPECT_THROW(wal.check_ok(), std::runtime_error);
}

TEST(WriteAheadLog, ShortWriteStopsTheLog) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::Group}, false);
    const auto lsn = wal.append(WalRecordType::Insert, bytes("torn"));
    fs.tear_next_append();
    EXPECT_THROW(wal.wait(lsn), std::runtime_error);
    EXPECT_THROW(wal.wait(lsn), std::runtime_error);
    EXPECT_TRUE(read_dir(fs, "/db").empty());  // the half record is not readable
}

TEST(WriteAheadLog, EveryWriteSyncsEachRecordAlone) {
    MemoryFileSystem fs;
    fs.make_dir("/db");
    WriteAheadLog wal(fs, "/db", 1, {SyncMode::EveryWrite}, false);
    {
        std::vector<std::jthread> threads;
        threads.reserve(4);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&wal] {
                for (int i = 0; i < 50; ++i) {
                    wal.wait(wal.append(WalRecordType::Insert, bytes("r")));
                }
            });
        }
    }
    EXPECT_EQ(wal.record_count(), 200u);
    EXPECT_EQ(wal.sync_count(), 200u);
    EXPECT_EQ(read_dir(fs, "/db").size(), 200u);
}
