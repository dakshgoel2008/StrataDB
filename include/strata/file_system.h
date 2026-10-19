#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata {

// A file being written. It is written from the start and grows at the end.
class WritableFile {
public:
    virtual ~WritableFile() = default;
    virtual void append(const void* data, std::size_t size) = 0;
    // Overwrites bytes that were already appended.
    virtual void write_at(std::uint64_t offset, const void* data, std::size_t size) = 0;
    // Returns once everything written so far is on disk (fdatasync).
    virtual void sync() = 0;
    virtual std::uint64_t size() const = 0;
};

class ReadableFile {
public:
    virtual ~ReadableFile() = default;
    virtual std::uint64_t size() const = 0;
    // Reads exactly `size` bytes starting at offset. Throws if the file is shorter.
    virtual void read(std::uint64_t offset, void* out, std::size_t size) const = 0;
};

// Held while a database directory is open, so two programs can't use it at once.
class FileLock {
public:
    virtual ~FileLock() = default;
};

// Every disk operation of the storage code goes through this interface, so tests
// can use a fake disk that fails or loses unsynced data on purpose.
// All errors are reported by throwing std::runtime_error.
class FileSystem {
public:
    virtual ~FileSystem() = default;
    // Creates an empty file for writing, replacing any file with that name.
    virtual std::unique_ptr<WritableFile> create(const std::string& path) = 0;
    // Opens an existing file to append to its end.
    virtual std::unique_ptr<WritableFile> open_append(const std::string& path) = 0;
    virtual std::unique_ptr<ReadableFile> open_read(const std::string& path) = 0;
    virtual bool exists(const std::string& path) = 0;
    // Names (not full paths) of the files in a directory.
    virtual std::vector<std::string> list(const std::string& dir) = 0;
    // Creates the directory if it doesn't exist yet.
    virtual void make_dir(const std::string& dir) = 0;
    virtual void rename(const std::string& from, const std::string& to) = 0;
    virtual void remove(const std::string& path) = 0;
    // Cuts a file down to `size` bytes and syncs it.
    virtual void truncate(const std::string& path, std::uint64_t size) = 0;
    // Makes file creations, renames and removals in dir survive a system crash.
    virtual void sync_dir(const std::string& dir) = 0;
    // Throws if another program holds the lock.
    virtual std::unique_ptr<FileLock> lock(const std::string& path) = 0;
};

// The real disk.
FileSystem& local_file_system();

std::string join_path(const std::string& dir, const std::string& name);
std::string parent_dir(const std::string& path);

std::vector<std::uint8_t> read_whole_file(FileSystem& fs, const std::string& path);

// Replaces a small file so that after a crash there is either the old file or
// the complete new one: write a temp file, sync it, rename it, sync the directory.
void write_file_atomically(FileSystem& fs, const std::string& path, const void* data,
                           std::size_t size);

// A disk kept in memory, for tests and fuzzing. It can also fail on purpose.
class MemoryFileSystem : public FileSystem {
public:
    MemoryFileSystem();
    ~MemoryFileSystem() override;

    std::unique_ptr<WritableFile> create(const std::string& path) override;
    std::unique_ptr<WritableFile> open_append(const std::string& path) override;
    std::unique_ptr<ReadableFile> open_read(const std::string& path) override;
    bool exists(const std::string& path) override;
    std::vector<std::string> list(const std::string& dir) override;
    void make_dir(const std::string& dir) override;
    void rename(const std::string& from, const std::string& to) override;
    void remove(const std::string& path) override;
    void truncate(const std::string& path, std::uint64_t size) override;
    void sync_dir(const std::string& dir) override;
    std::unique_ptr<FileLock> lock(const std::string& path) override;

    // The next sync fails with an error, but the disk keeps working.
    void fail_next_sync();
    // The next append writes only part of its data, then fails.
    void tear_next_append();

    // Whole contents of a file, for tests that want to damage it on purpose.
    std::vector<std::uint8_t>& raw_contents(const std::string& path);

private:
    using Data = std::shared_ptr<std::vector<std::uint8_t>>;
    class Writer;
    class Reader;

    Data find(const std::string& path) const;

    mutable std::mutex mutex_;
    std::map<std::string, Data> files_;
    std::vector<std::string> dirs_;
    bool fail_sync_ = false;
    bool tear_append_ = false;
};

}  // namespace strata
