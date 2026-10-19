#include "strata/wal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "checksum_policy.h"
#include "strata/crc32c.h"

namespace strata {
namespace {

std::uint32_t record_crc(const std::uint8_t* header, std::span<const std::uint8_t> payload) {
    std::uint32_t crc = crc32c(header, 4);               // the length
    crc = crc32c(header + 8, kWalHeaderBytes - 8, crc);  // lsn and type
    return crc32c(payload.data(), payload.size(), crc);
}

constexpr char kPrefix[] = "wal_";
constexpr char kSuffix[] = ".log";
constexpr std::size_t kDigits = 20;

}  // namespace

void encode_wal_record(std::uint64_t lsn, WalRecordType type, std::span<const std::uint8_t> payload,
                       std::vector<std::uint8_t>& out) {
    if (payload.size() > kMaxWalPayloadBytes) {
        throw std::invalid_argument("log record is too large");
    }
    std::uint8_t header[kWalHeaderBytes];
    const auto length = static_cast<std::uint32_t>(payload.size());
    std::memcpy(header, &length, 4);
    std::memcpy(header + 8, &lsn, 8);
    header[16] = static_cast<std::uint8_t>(type);
    const std::uint32_t crc = record_crc(header, payload);
    std::memcpy(header + 4, &crc, 4);
    out.insert(out.end(), header, header + kWalHeaderBytes);
    out.insert(out.end(), payload.begin(), payload.end());
}

std::size_t read_wal_records(std::span<const std::uint8_t> data, std::uint64_t first_lsn,
                             const std::function<void(const WalRecord&)>& fn) {
    std::size_t pos = 0;
    std::uint64_t expected = first_lsn;
    while (data.size() - pos >= kWalHeaderBytes) {
        const std::uint8_t* header = data.data() + pos;
        std::uint32_t length;
        std::uint32_t crc;
        WalRecord record;
        std::memcpy(&length, header, 4);
        std::memcpy(&crc, header + 4, 4);
        std::memcpy(&record.lsn, header + 8, 8);
        if (length > kMaxWalPayloadBytes || data.size() - pos - kWalHeaderBytes < length) {
            break;  // cut off, or the length itself is damaged
        }
        record.payload = data.subspan(pos + kWalHeaderBytes, length);
        if ((kVerifyChecksums && record_crc(header, record.payload) != crc) ||
            record.lsn != expected) {
            break;
        }
        record.type = static_cast<WalRecordType>(header[16]);
        fn(record);
        pos += kWalHeaderBytes + length;
        ++expected;
    }
    return pos;
}

std::string wal_file_name(std::uint64_t first_lsn) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s%020llu%s", kPrefix,
                  static_cast<unsigned long long>(first_lsn), kSuffix);
    return name;
}

bool parse_wal_file_name(const std::string& name, std::uint64_t& first_lsn) {
    const std::size_t prefix = sizeof(kPrefix) - 1;
    const std::size_t suffix = sizeof(kSuffix) - 1;
    if (name.size() != prefix + kDigits + suffix || name.compare(0, prefix, kPrefix) != 0 ||
        name.compare(prefix + kDigits, suffix, kSuffix) != 0) {
        return false;
    }
    std::uint64_t value = 0;
    for (std::size_t i = prefix; i < prefix + kDigits; ++i) {
        const char c = name[i];
        if (c < '0' || c > '9') {
            return false;
        }
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (value > (~std::uint64_t{0} - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    first_lsn = value;
    return true;
}

const char* sync_mode_name(SyncMode mode) noexcept {
    switch (mode) {
        case SyncMode::EveryWrite:
            return "every-write";
        case SyncMode::Group:
            return "group";
        case SyncMode::None:
            return "none";
    }
    return "?";
}

SyncMode parse_sync_mode(const std::string& name) {
    for (SyncMode m : {SyncMode::EveryWrite, SyncMode::Group, SyncMode::None}) {
        if (name == sync_mode_name(m)) {
            return m;
        }
    }
    throw std::invalid_argument("unknown sync mode '" + name + "'");
}

WriteAheadLog::WriteAheadLog(FileSystem& fs, std::string dir, std::uint64_t next_lsn,
                             Options options, bool append_to_last)
    : fs_(fs),
      dir_(std::move(dir)),
      options_(options),
      next_lsn_(next_lsn),
      written_lsn_(next_lsn - 1) {
    if (next_lsn == 0) {
        throw std::invalid_argument("log numbers start at 1");
    }
    if (append_to_last) {
        std::uint64_t newest = 0;
        bool found = false;
        for (const std::string& name : fs_.list(dir_)) {
            std::uint64_t first = 0;
            if (parse_wal_file_name(name, first) && (!found || first > newest)) {
                newest = first;
                found = true;
            }
        }
        if (found) {
            file_ = fs_.open_append(join_path(dir_, wal_file_name(newest)));
            file_first_lsn_ = newest;
            return;
        }
    }
    open_new_file(next_lsn);
}

WriteAheadLog::~WriteAheadLog() = default;

void WriteAheadLog::open_new_file(std::uint64_t first_lsn) {
    if (file_ && file_first_lsn_ == first_lsn) {
        return;  // the current file is still empty
    }
    if (file_) {
        file_->sync();
        ++syncs_;
    }
    file_ = fs_.create(join_path(dir_, wal_file_name(first_lsn)));
    file_first_lsn_ = first_lsn;
    // The new name must be on disk before any record that depends on it.
    fs_.sync_dir(dir_);
}

void WriteAheadLog::write_batch(const std::vector<std::uint8_t>& batch, std::uint64_t batch_last) {
    if (!batch.empty()) {
        file_->append(batch.data(), batch.size());
        if (options_.sync != SyncMode::None) {
            file_->sync();
            ++syncs_;
        }
    }
    if (file_->size() >= options_.file_bytes) {
        open_new_file(batch_last + 1);
    }
}

std::uint64_t WriteAheadLog::append(WalRecordType type, std::span<const std::uint8_t> payload) {
    std::unique_lock lock(mutex_);
    if (options_.sync == SyncMode::EveryWrite) {
        // Becomes the only writer first, so this record gets a sync of its own.
        begin_flush(lock);
        const std::uint64_t lsn = next_lsn_++;
        ++records_;
        lock.unlock();
        std::vector<std::uint8_t> record;
        try {
            encode_wal_record(lsn, type, payload, record);
            write_batch(record, lsn);
        } catch (...) {
            end_flush(lock, false, 0);
            throw;
        }
        end_flush(lock, true, lsn);
        return lsn;
    }
    if (failed_) {
        throw std::runtime_error("the log stopped after a write error");
    }
    const std::uint64_t lsn = next_lsn_;
    encode_wal_record(lsn, type, payload, buffer_);
    ++next_lsn_;
    ++records_;
    return lsn;
}

void WriteAheadLog::begin_flush(std::unique_lock<std::mutex>& lock) {
    while (flushing_ && !failed_) {
        flushed_.wait(lock);
    }
    if (failed_) {
        throw std::runtime_error("the log stopped after a write error");
    }
    flushing_ = true;
}

void WriteAheadLog::end_flush(std::unique_lock<std::mutex>& lock, bool ok, std::uint64_t written) {
    lock.lock();
    flushing_ = false;
    if (ok) {
        written_lsn_ = written;
    } else {
        failed_ = true;
    }
    flushed_.notify_all();
}

// One waiting thread becomes the writer and takes every record queued so far.
// Threads that arrive while it is busy queue up behind it, and the next writer
// takes all of their records in one go. That is the group commit.
void WriteAheadLog::wait(std::uint64_t lsn) {
    std::unique_lock lock(mutex_);
    while (written_lsn_ < lsn && flushing_ && !failed_) {
        flushed_.wait(lock);
    }
    if (failed_) {
        throw std::runtime_error("the log stopped after a write error");
    }
    if (written_lsn_ >= lsn) {
        return;
    }
    begin_flush(lock);
    std::vector<std::uint8_t> batch;
    batch.swap(buffer_);
    const std::uint64_t batch_last = next_lsn_ - 1;
    lock.unlock();
    try {
        write_batch(batch, batch_last);
    } catch (...) {
        end_flush(lock, false, 0);
        throw;
    }
    end_flush(lock, true, batch_last);
}

std::uint64_t WriteAheadLog::last_lsn() const {
    std::lock_guard lock(mutex_);
    return next_lsn_ - 1;
}

void WriteAheadLog::start_new_file() {
    std::unique_lock lock(mutex_);
    begin_flush(lock);
    std::vector<std::uint8_t> batch;
    batch.swap(buffer_);
    const std::uint64_t batch_last = next_lsn_ - 1;
    lock.unlock();
    try {
        write_batch(batch, batch_last);
        open_new_file(batch_last + 1);
    } catch (...) {
        end_flush(lock, false, 0);
        throw;
    }
    end_flush(lock, true, batch_last);
}

void WriteAheadLog::remove_files_through(std::uint64_t lsn) {
    std::vector<std::uint64_t> firsts;
    for (const std::string& name : fs_.list(dir_)) {
        std::uint64_t first = 0;
        if (parse_wal_file_name(name, first)) {
            firsts.push_back(first);
        }
    }
    std::sort(firsts.begin(), firsts.end());
    bool removed = false;
    // A file ends where the next one starts, and the newest file is never removed.
    for (std::size_t i = 0; i + 1 < firsts.size(); ++i) {
        if (firsts[i + 1] <= lsn + 1) {
            fs_.remove(join_path(dir_, wal_file_name(firsts[i])));
            removed = true;
        }
    }
    if (removed) {
        fs_.sync_dir(dir_);
    }
}

void WriteAheadLog::check_ok() const {
    std::lock_guard lock(mutex_);
    if (failed_) {
        throw std::runtime_error("the log stopped after a write error");
    }
}

}  // namespace strata
