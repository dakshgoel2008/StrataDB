#include "strata/database.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "checksum_policy.h"
#include "strata/crc32c.h"

namespace strata {
namespace {

constexpr char kManifestMagic[8] = {'S', 'T', 'R', 'M', 'A', 'N', 'I', '1'};
constexpr std::uint32_t kManifestVersion = 1;
constexpr char kManifestName[] = "MANIFEST";
constexpr char kLockName[] = "LOCK";
constexpr std::size_t kInsertHeaderBytes = 16;
constexpr std::size_t kRecordTargetBytes = 4u
                                           << 20;  // big inserts are split into records this size
constexpr std::size_t kReplayBatchRows = 8192;

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

bool ends_with(const std::string& s, const std::string& end) {
    return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

std::vector<std::uint64_t> sorted_wal_files(FileSystem& fs, const std::string& dir) {
    std::vector<std::uint64_t> firsts;
    for (const std::string& name : fs.list(dir)) {
        std::uint64_t first = 0;
        if (parse_wal_file_name(name, first)) {
            firsts.push_back(first);
        }
    }
    std::sort(firsts.begin(), firsts.end());
    return firsts;
}

}  // namespace

std::array<std::uint8_t, kManifestBytes> encode_manifest(const Manifest& m) {
    std::array<std::uint8_t, kManifestBytes> out{};
    std::memcpy(out.data(), kManifestMagic, 8);
    std::memcpy(out.data() + 8, &kManifestVersion, 4);
    std::memcpy(out.data() + 16, &m.snapshot_id, 8);
    std::memcpy(out.data() + 24, &m.wal_lsn, 8);
    const std::uint32_t crc = crc32c(out.data(), 32);
    std::memcpy(out.data() + 32, &crc, 4);
    return out;
}

Manifest parse_manifest(std::span<const std::uint8_t> bytes) {
    if (bytes.size() != kManifestBytes || std::memcmp(bytes.data(), kManifestMagic, 8) != 0) {
        throw std::runtime_error("MANIFEST is not a StrataDB manifest");
    }
    std::uint32_t version;
    std::uint32_t crc;
    std::memcpy(&version, bytes.data() + 8, 4);
    std::memcpy(&crc, bytes.data() + 32, 4);
    if (version != kManifestVersion || (kVerifyChecksums && crc != crc32c(bytes.data(), 32))) {
        throw std::runtime_error("MANIFEST is damaged or from an unknown version");
    }
    Manifest m;
    std::memcpy(&m.snapshot_id, bytes.data() + 16, 8);
    std::memcpy(&m.wal_lsn, bytes.data() + 24, 8);
    if (m.snapshot_id == 0) {
        throw std::runtime_error("MANIFEST names no snapshot");
    }
    return m;
}

std::vector<std::uint8_t> encode_insert(std::uint64_t first_row, std::span<const float> vectors,
                                        std::size_t dim) {
    const auto count = static_cast<std::uint32_t>(vectors.size() / dim);
    const auto dim32 = static_cast<std::uint32_t>(dim);
    std::vector<std::uint8_t> out(kInsertHeaderBytes + vectors.size_bytes());
    std::memcpy(out.data(), &first_row, 8);
    std::memcpy(out.data() + 8, &count, 4);
    std::memcpy(out.data() + 12, &dim32, 4);
    std::memcpy(out.data() + kInsertHeaderBytes, vectors.data(), vectors.size_bytes());
    return out;
}

bool decode_insert(std::span<const std::uint8_t> payload, InsertRecord& out) {
    if (payload.size() < kInsertHeaderBytes) {
        return false;
    }
    std::memcpy(&out.first_row, payload.data(), 8);
    std::memcpy(&out.count, payload.data() + 8, 4);
    std::memcpy(&out.dim, payload.data() + 12, 4);
    const std::uint64_t bytes = std::uint64_t{out.count} * out.dim * sizeof(float);
    if (out.count == 0 || out.dim == 0 || payload.size() - kInsertHeaderBytes != bytes) {
        return false;
    }
    out.vectors = payload.subspan(kInsertHeaderBytes);
    return true;
}

std::string snapshot_file_name(std::uint64_t id) {
    char name[64];
    std::snprintf(name, sizeof(name), "snapshot_%020llu.snap", static_cast<unsigned long long>(id));
    return name;
}

bool parse_snapshot_file_name(const std::string& name, std::uint64_t& id) {
    constexpr std::size_t kPrefix = 9;  // "snapshot_"
    constexpr std::size_t kDigits = 20;
    if (name.size() != kPrefix + kDigits + 5 || name.compare(0, kPrefix, "snapshot_") != 0 ||
        !ends_with(name, ".snap")) {
        return false;
    }
    std::uint64_t value = 0;
    for (std::size_t i = kPrefix; i < kPrefix + kDigits; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
        value = value * 10 + static_cast<std::uint64_t>(name[i] - '0');
    }
    id = value;
    return true;
}

Database::Database(FileSystem& fs, std::string dir, const DatabaseOptions& options)
    : fs_(fs), dir_(std::move(dir)), options_(options) {
    fs_.make_dir(dir_);
    lock_ = fs_.lock(path(kLockName));
}

Database::~Database() = default;

std::string Database::path(const std::string& name) const { return join_path(dir_, name); }

std::unique_ptr<Database> Database::create(const std::string& dir, std::unique_ptr<HnswIndex> index,
                                           const DatabaseOptions& options) {
    if (!index) {
        throw std::invalid_argument("create() needs an index");
    }
    FileSystem& fs = options.fs != nullptr ? *options.fs : local_file_system();
    std::unique_ptr<Database> db(new Database(fs, dir, options));
    if (fs.exists(db->path(kManifestName))) {
        throw std::runtime_error("'" + dir + "' already holds a database");
    }
    // Files left by an earlier create() that crashed before writing MANIFEST.
    db->remove_leftovers(0, true);

    db->index_ = std::move(index);
    db->snapshot_id_ = 1;
    db->next_snapshot_id_ = 2;
    db->index_->save_snapshot(fs, db->path(snapshot_file_name(1)), db->index_->save_point(), 0);
    const auto manifest = encode_manifest({1, 0});
    write_file_atomically(fs, db->path(kManifestName), manifest.data(), manifest.size());
    db->wal_ = std::make_unique<WriteAheadLog>(
        fs, dir, 1, WriteAheadLog::Options{options.sync, options.wal_file_bytes}, false);
    return db;
}

std::unique_ptr<Database> Database::open(const std::string& dir, const DatabaseOptions& options) {
    FileSystem& fs = options.fs != nullptr ? *options.fs : local_file_system();
    std::unique_ptr<Database> db(new Database(fs, dir, options));
    if (!fs.exists(db->path(kManifestName))) {
        throw std::runtime_error("'" + dir + "' holds no database");
    }
    const Manifest manifest = parse_manifest(read_whole_file(fs, db->path(kManifestName)));
    db->remove_leftovers(manifest.snapshot_id, false);
    db->recover(manifest);
    return db;
}

void Database::remove_leftovers(std::uint64_t keep_snapshot, bool remove_logs) {
    for (const std::string& name : fs_.list(dir_)) {
        std::uint64_t number = 0;
        const bool old_snapshot = parse_snapshot_file_name(name, number) && number != keep_snapshot;
        const bool log = remove_logs && parse_wal_file_name(name, number);
        if (old_snapshot || log || ends_with(name, ".tmp")) {
            fs_.remove(path(name));
        }
    }
}

void Database::recover(const Manifest& manifest) {
    auto start = std::chrono::steady_clock::now();
    auto loaded = HnswIndex::load_snapshot(fs_, path(snapshot_file_name(manifest.snapshot_id)),
                                           options_.capacity);
    if (loaded.wal_lsn != manifest.wal_lsn) {
        throw std::runtime_error("snapshot and MANIFEST disagree about the log position");
    }
    index_ = std::move(loaded.index);
    snapshot_id_ = manifest.snapshot_id;
    next_snapshot_id_ = manifest.snapshot_id + 1;
    recovery_.snapshot_id = manifest.snapshot_id;
    recovery_.snapshot_rows = index_->size();
    recovery_.snapshot_lsn = manifest.wal_lsn;
    recovery_.load_seconds = seconds_since(start);
    start = std::chrono::steady_clock::now();

    // Files that end before the snapshot's position are not needed any more.
    std::vector<std::uint64_t> files = sorted_wal_files(fs_, dir_);
    std::size_t first_needed = 0;
    while (first_needed + 1 < files.size() && files[first_needed + 1] <= manifest.wal_lsn + 1) {
        fs_.remove(path(wal_file_name(files[first_needed])));
        ++first_needed;
    }
    files.erase(files.begin(), files.begin() + static_cast<long>(first_needed));
    if (!files.empty() && files.front() > manifest.wal_lsn + 1) {
        throw std::runtime_error("log records right after the snapshot are missing");
    }

    const std::size_t dim = index_->dim();
    std::vector<float> pending;  // replayed vectors not added to the index yet
    std::uint64_t pending_rows = 0;
    auto add_pending = [&] {
        if (pending.empty()) {
            return;
        }
        if (index_->size() + pending_rows > index_->capacity()) {
            throw std::runtime_error(
                "the log holds more rows than the capacity; open with a larger capacity");
        }
        index_->add(pending, options_.replay_threads);
        pending.clear();
        pending_rows = 0;
    };

    std::uint64_t next_lsn = files.empty() ? manifest.wal_lsn + 1 : files.front();
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (files[i] != next_lsn) {
            throw std::runtime_error("log files have a gap before " + wal_file_name(files[i]));
        }
        const std::string file_path = path(wal_file_name(files[i]));
        const std::vector<std::uint8_t> data = read_whole_file(fs_, file_path);
        const std::size_t good = read_wal_records(data, files[i], [&](const WalRecord& r) {
            ++next_lsn;
            if (r.lsn <= manifest.wal_lsn) {
                return;  // already in the snapshot
            }
            InsertRecord ins;
            if (r.type != WalRecordType::Insert || !decode_insert(r.payload, ins) ||
                ins.dim != dim || ins.first_row != index_->size() + pending_rows) {
                throw std::runtime_error("log record " + std::to_string(r.lsn) +
                                         " doesn't fit the index");
            }
            const std::size_t old_size = pending.size();
            pending.resize(old_size + std::size_t{ins.count} * dim);
            std::memcpy(pending.data() + old_size, ins.vectors.data(), ins.vectors.size());
            pending_rows += ins.count;
            ++recovery_.replayed_records;
            recovery_.replayed_rows += ins.count;
            if (pending_rows >= kReplayBatchRows) {
                add_pending();
            }
        });
        if (good < data.size()) {
            // Only the newest file can end in a half-written record; anywhere else it is damage.
            if (i + 1 < files.size()) {
                throw std::runtime_error(wal_file_name(files[i]) + " is damaged before its end");
            }
            fs_.truncate(file_path, good);
            recovery_.dropped_bytes = data.size() - good;
        }
    }
    add_pending();

    // If the log ends before the snapshot (unsynced records lost in a system crash
    // with sync mode None), new records start a fresh file after the snapshot.
    const bool continues_log = !files.empty() && next_lsn > manifest.wal_lsn;
    next_lsn = std::max(next_lsn, manifest.wal_lsn + 1);
    wal_ = std::make_unique<WriteAheadLog>(
        fs_, dir_, next_lsn, WriteAheadLog::Options{options_.sync, options_.wal_file_bytes},
        continues_log);
    recovery_.replay_seconds = seconds_since(start);
}

void Database::throw_if_failed() const {
    if (failed_.load()) {
        throw std::runtime_error("database stopped after a disk error; reopen it to recover");
    }
}

RowId Database::insert(std::span<const float> vectors) {
    const std::size_t dim = index_->dim();
    if (vectors.size() % dim != 0) {
        throw std::invalid_argument("vector data size is not a multiple of dim");
    }
    const std::size_t n = vectors.size() / dim;
    if (n == 0) {
        return static_cast<RowId>(index_->size());
    }

    // Large inserts become several records, so no single record is huge.
    const std::size_t rows_per_record = std::max<std::size_t>(1, kRecordTargetBytes / (dim * 4));
    std::vector<std::vector<std::uint8_t>> payloads;
    for (std::size_t row = 0; row < n; row += rows_per_record) {
        const std::size_t rows = std::min(rows_per_record, n - row);
        payloads.push_back(encode_insert(0, vectors.subspan(row * dim, rows * dim), dim));
    }

    std::unique_lock gate(gate_);
    std::shared_lock pause(pause_);
    gate.unlock();
    throw_if_failed();

    RowId first = 0;
    std::uint64_t last_lsn = 0;
    {
        std::lock_guard order(order_);
        throw_if_failed();
        first = index_->reserve(n);
        try {
            for (std::size_t i = 0; i < payloads.size(); ++i) {
                const std::uint64_t row = static_cast<std::uint64_t>(first) + i * rows_per_record;
                std::memcpy(payloads[i].data(), &row, 8);
                last_lsn = wal_->append(WalRecordType::Insert, payloads[i]);
            }
        } catch (...) {
            failed_ = true;  // the reserved rows will never be filled
            throw;
        }
    }
    try {
        wal_->wait(last_lsn);
    } catch (...) {
        failed_ = true;
        throw;
    }
    index_->insert_reserved(first, vectors);
    return first;
}

void Database::checkpoint() {
    std::lock_guard one_at_a_time(checkpoint_lock_);
    throw_if_failed();

    HnswIndex::SavePoint point;
    std::uint64_t lsn = 0;
    {
        // Wait for running inserts, so the index holds exactly the records up to lsn.
        std::lock_guard gate(gate_);
        std::unique_lock pause(pause_);
        point = index_->save_point();
        lsn = wal_->last_lsn();
        try {
            wal_->start_new_file();
        } catch (...) {
            failed_ = true;
            throw;
        }
    }

    // Inserts run again from here while the snapshot is written.
    const std::uint64_t id = next_snapshot_id_++;
    index_->save_snapshot(fs_, path(snapshot_file_name(id)), point, lsn);
    const auto manifest = encode_manifest({id, lsn});
    write_file_atomically(fs_, path(kManifestName), manifest.data(), manifest.size());

    // The new snapshot is now the one that counts.
    const std::uint64_t old = snapshot_id_;
    snapshot_id_ = id;
    fs_.remove(path(snapshot_file_name(old)));
    wal_->remove_files_through(lsn);
}

}  // namespace strata
