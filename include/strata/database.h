#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#include "strata/file_system.h"
#include "strata/hnsw_index.h"
#include "strata/wal.h"

namespace strata {

struct DatabaseOptions {
    SyncMode sync = SyncMode::Group;
    std::uint64_t wal_file_bytes = 64u << 20;  // start a new log file after this size
    std::size_t capacity = 0;                  // when opening: 0 keeps the saved capacity
    std::size_t replay_threads = 0;            // threads used to replay the log (0 = all cores)
    FileSystem* fs = nullptr;                  // nullptr = the local disk
};

// What open() found and did.
struct RecoveryInfo {
    std::uint64_t snapshot_id = 0;
    std::size_t snapshot_rows = 0;
    std::uint64_t snapshot_lsn = 0;
    std::uint64_t replayed_records = 0;
    std::size_t replayed_rows = 0;
    std::uint64_t dropped_bytes = 0;  // damaged or cut-off bytes removed from the end of the log
    double load_seconds = 0;
    double replay_seconds = 0;
};

// The MANIFEST file names the current snapshot and the last log record it includes.
struct Manifest {
    std::uint64_t snapshot_id = 0;
    std::uint64_t wal_lsn = 0;
};
inline constexpr std::size_t kManifestBytes = 40;
std::array<std::uint8_t, kManifestBytes> encode_manifest(const Manifest& m);
// Throws std::runtime_error if the bytes are not a valid manifest.
Manifest parse_manifest(std::span<const std::uint8_t> bytes);

// Payload of an Insert log record: u64 first row | u32 count | u32 dim | count * dim floats.
struct InsertRecord {
    std::uint64_t first_row = 0;
    std::uint32_t count = 0;
    std::uint32_t dim = 0;
    std::span<const std::uint8_t> vectors;  // count * dim floats, maybe not aligned
};
std::vector<std::uint8_t> encode_insert(std::uint64_t first_row, std::span<const float> vectors,
                                        std::size_t dim);
// Returns false if the payload is not a well-formed insert.
bool decode_insert(std::span<const std::uint8_t> payload, InsertRecord& out);

// An HNSW index that survives crashes.
//
// Every insert is written to the log before it is applied. checkpoint() saves a
// snapshot of the whole index and drops the log it covers. open() loads the
// newest snapshot and replays the log after it.
//
// Files in the directory:
//   MANIFEST                 which snapshot is current
//   snapshot_<id>.snap       the index as of some log record
//   wal_<first lsn>.log      log records
//   LOCK                     held while the database is open
class Database {
public:
    // Creates a new database in dir (made if missing; must not hold a database).
    // It starts from `index`, which is usually empty but may already hold vectors.
    static std::unique_ptr<Database> create(const std::string& dir,
                                            std::unique_ptr<HnswIndex> index,
                                            const DatabaseOptions& options = {});

    // Opens an existing database and brings it back to its last saved state.
    static std::unique_ptr<Database> open(const std::string& dir,
                                          const DatabaseOptions& options = {});

    ~Database();
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    // Adds vectors (n * dim floats) and returns the row id of the first one.
    // When it returns, the vectors are in the log (synced as the sync mode
    // promises) and can be found by search. Safe to call from many threads.
    // After a disk error every later insert throws; reopen to recover.
    RowId insert(std::span<const float> vectors);

    void search(std::span<const float> query, std::size_t k, std::size_t ef_search,
                std::span<RowId> ids, std::span<float> distances) const {
        index_->search(query, k, ef_search, ids, distances);
    }

    // Saves a snapshot and deletes the log files and old snapshot it replaces.
    // Inserts pause only while the ones in progress finish; searches never pause.
    void checkpoint();

    const HnswIndex& index() const noexcept { return *index_; }
    std::size_t size() const noexcept { return index_->size(); }
    const RecoveryInfo& recovery() const noexcept { return recovery_; }
    const WriteAheadLog& wal() const noexcept { return *wal_; }

private:
    Database(FileSystem& fs, std::string dir, const DatabaseOptions& options);

    std::string path(const std::string& name) const;
    void recover(const Manifest& manifest);
    // Removes temp files and snapshots other than keep_snapshot (and log files if asked).
    void remove_leftovers(std::uint64_t keep_snapshot, bool remove_logs);
    void throw_if_failed() const;

    FileSystem& fs_;
    std::string dir_;
    DatabaseOptions options_;
    std::unique_ptr<FileLock> lock_;
    std::unique_ptr<HnswIndex> index_;
    std::unique_ptr<WriteAheadLog> wal_;
    RecoveryInfo recovery_;

    // Inserts hold pause_ shared; checkpoint() takes it alone for a moment.
    // gate_ stops new inserts from slipping in while checkpoint() waits for it.
    std::mutex gate_;
    std::shared_mutex pause_;
    // Row ids and log numbers are handed out together under this lock, so
    // replaying the log in lsn order gives every row its original id.
    std::mutex order_;
    std::mutex checkpoint_lock_;
    std::uint64_t snapshot_id_ = 0;       // the one MANIFEST names
    std::uint64_t next_snapshot_id_ = 0;  // ids are never reused, even after a failed checkpoint
    std::atomic<bool> failed_{false};
};

std::string snapshot_file_name(std::uint64_t id);
bool parse_snapshot_file_name(const std::string& name, std::uint64_t& id);

}  // namespace strata
