#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "strata/file_system.h"

namespace strata {

// Write-ahead log. Every change is appended here before it is applied, so it
// can be replayed after a crash. Records are numbered 1, 2, 3, ... (the LSN).
//
// Record layout (little-endian):
//   u32 payload length | u32 CRC32C | u64 lsn | u8 type | payload
// The CRC covers the length, lsn, type and payload.
//
// The log is split into files named wal_<lsn of first record>.log. A new file
// is started when the current one gets too big, and at every checkpoint.

enum class WalRecordType : std::uint8_t {
    Insert = 1,
};

inline constexpr std::size_t kWalHeaderBytes = 17;
inline constexpr std::size_t kMaxWalPayloadBytes = 256u << 20;

struct WalRecord {
    std::uint64_t lsn = 0;
    WalRecordType type = WalRecordType::Insert;
    std::span<const std::uint8_t> payload;
};

// Appends one encoded record to out.
void encode_wal_record(std::uint64_t lsn, WalRecordType type, std::span<const std::uint8_t> payload,
                       std::vector<std::uint8_t>& out);

// Reads records from the start of data, calling fn for each one. The first must
// have lsn first_lsn and each next one lsn + 1. Stops at the end of the data or at
// the first record that is cut off, fails its CRC or has the wrong lsn.
// Returns how many bytes of good records were read.
std::size_t read_wal_records(std::span<const std::uint8_t> data, std::uint64_t first_lsn,
                             const std::function<void(const WalRecord&)>& fn);

std::string wal_file_name(std::uint64_t first_lsn);
// Returns false if name is not a log file name.
bool parse_wal_file_name(const std::string& name, std::uint64_t& first_lsn);

enum class SyncMode : std::uint8_t {
    EveryWrite,  // each record gets its own fdatasync
    Group,       // writers waiting at the same time share one fdatasync
    None,        // no fdatasync: survives a crashed program, not a system crash
};

const char* sync_mode_name(SyncMode mode) noexcept;
SyncMode parse_sync_mode(const std::string& name);

class WriteAheadLog {
public:
    struct Options {
        SyncMode sync = SyncMode::Group;
        std::uint64_t file_bytes = 64u << 20;  // start a new file after this size
    };

    // Opens the log in dir for writing. The next record gets next_lsn. With
    // append_to_last, records go to the end of the newest existing file (recovery
    // has already checked it ends at next_lsn - 1); otherwise a new file is started.
    WriteAheadLog(FileSystem& fs, std::string dir, std::uint64_t next_lsn, Options options,
                  bool append_to_last);
    ~WriteAheadLog();
    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    // Adds a record and returns its lsn. In EveryWrite mode it is written and
    // synced on its own before this returns; in the other modes call wait()
    // before relying on it.
    std::uint64_t append(WalRecordType type, std::span<const std::uint8_t> payload);

    // Returns once record lsn (and all before it) are written and, unless the mode
    // is None, synced. Threads that wait at the same time share one sync.
    void wait(std::uint64_t lsn);

    // Lsn of the last record given to append().
    std::uint64_t last_lsn() const;

    // Writes and syncs everything, then starts a new file.
    void start_new_file();

    // Deletes the log files whose records are all <= lsn (never the current file).
    void remove_files_through(std::uint64_t lsn);

    // Throws if an earlier write failed. After a failure nothing more is written.
    void check_ok() const;

    std::uint64_t sync_count() const noexcept { return syncs_.load(); }
    std::uint64_t record_count() const noexcept { return records_.load(); }

private:
    // Writes `batch` (records up to batch_last) to the current file. Only the
    // thread that set flushing_ calls this.
    void write_batch(const std::vector<std::uint8_t>& batch, std::uint64_t batch_last);
    void open_new_file(std::uint64_t first_lsn);
    // Becomes the only writer, waiting for any current one to finish.
    void begin_flush(std::unique_lock<std::mutex>& lock);
    void end_flush(std::unique_lock<std::mutex>& lock, bool ok, std::uint64_t written);

    FileSystem& fs_;
    std::string dir_;
    Options options_;

    mutable std::mutex mutex_;
    std::condition_variable flushed_;
    std::vector<std::uint8_t> buffer_;  // records not written yet
    std::uint64_t next_lsn_;
    std::uint64_t written_lsn_;  // last record written (and synced, unless mode is None)
    bool flushing_ = false;
    bool failed_ = false;

    // Only touched by the flushing thread.
    std::unique_ptr<WritableFile> file_;
    std::uint64_t file_first_lsn_ = 0;

    std::atomic<std::uint64_t> syncs_{0};
    std::atomic<std::uint64_t> records_{0};
};

}  // namespace strata
