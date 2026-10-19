#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "strata/file_system.h"

namespace strata {

class MemoryFileSystem::Writer final : public WritableFile {
public:
    Writer(MemoryFileSystem& fs, Data data) : fs_(fs), data_(std::move(data)) {}

    void append(const void* data, std::size_t size) override {
        std::lock_guard lock(fs_.mutex_);
        const auto* p = static_cast<const std::uint8_t*>(data);
        if (fs_.tear_append_) {
            fs_.tear_append_ = false;
            data_->insert(data_->end(), p, p + size / 2);
            throw std::runtime_error("simulated short write");
        }
        data_->insert(data_->end(), p, p + size);
    }

    void write_at(std::uint64_t offset, const void* data, std::size_t size) override {
        std::lock_guard lock(fs_.mutex_);
        if (offset + size > data_->size()) {
            data_->resize(offset + size);
        }
        if (size > 0) {
            std::memcpy(data_->data() + offset, data, size);
        }
    }

    void sync() override {
        std::lock_guard lock(fs_.mutex_);
        if (fs_.fail_sync_) {
            fs_.fail_sync_ = false;
            throw std::runtime_error("simulated sync failure");
        }
    }

    std::uint64_t size() const override {
        std::lock_guard lock(fs_.mutex_);
        return data_->size();
    }

private:
    MemoryFileSystem& fs_;
    Data data_;
};

class MemoryFileSystem::Reader final : public ReadableFile {
public:
    explicit Reader(std::vector<std::uint8_t> data) : data_(std::move(data)) {}

    std::uint64_t size() const override { return data_.size(); }

    void read(std::uint64_t offset, void* out, std::size_t size) const override {
        if (offset > data_.size() || size > data_.size() - offset) {
            throw std::runtime_error("read past the end of a file");
        }
        if (size > 0) {
            std::memcpy(out, data_.data() + offset, size);
        }
    }

private:
    std::vector<std::uint8_t> data_;  // a copy taken at open time
};

namespace {

class NoLock final : public FileLock {};

}  // namespace

MemoryFileSystem::MemoryFileSystem() = default;

MemoryFileSystem::~MemoryFileSystem() = default;

MemoryFileSystem::Data MemoryFileSystem::find(const std::string& path) const {
    const auto it = files_.find(path);
    if (it == files_.end()) {
        throw std::runtime_error("no such file '" + path + "'");
    }
    return it->second;
}

std::unique_ptr<WritableFile> MemoryFileSystem::create(const std::string& path) {
    std::lock_guard lock(mutex_);
    auto data = std::make_shared<std::vector<std::uint8_t>>();
    files_[path] = data;
    return std::make_unique<Writer>(*this, data);
}

std::unique_ptr<WritableFile> MemoryFileSystem::open_append(const std::string& path) {
    std::lock_guard lock(mutex_);
    return std::make_unique<Writer>(*this, find(path));
}

std::unique_ptr<ReadableFile> MemoryFileSystem::open_read(const std::string& path) {
    std::lock_guard lock(mutex_);
    return std::make_unique<Reader>(*find(path));
}

bool MemoryFileSystem::exists(const std::string& path) {
    std::lock_guard lock(mutex_);
    return files_.contains(path) || std::find(dirs_.begin(), dirs_.end(), path) != dirs_.end();
}

std::vector<std::string> MemoryFileSystem::list(const std::string& dir) {
    std::lock_guard lock(mutex_);
    std::vector<std::string> names;
    for (const auto& [path, data] : files_) {
        if (parent_dir(path) == dir) {
            names.push_back(path.substr(path.find_last_of('/') + 1));
        }
    }
    return names;
}

void MemoryFileSystem::make_dir(const std::string& dir) {
    std::lock_guard lock(mutex_);
    if (std::find(dirs_.begin(), dirs_.end(), dir) == dirs_.end()) {
        dirs_.push_back(dir);
    }
}

void MemoryFileSystem::rename(const std::string& from, const std::string& to) {
    std::lock_guard lock(mutex_);
    Data data = find(from);
    files_.erase(from);
    files_[to] = std::move(data);
}

void MemoryFileSystem::remove(const std::string& path) {
    std::lock_guard lock(mutex_);
    files_.erase(path);
}

void MemoryFileSystem::truncate(const std::string& path, std::uint64_t size) {
    std::lock_guard lock(mutex_);
    const Data data = find(path);
    data->resize(std::min<std::uint64_t>(size, data->size()));
}

void MemoryFileSystem::sync_dir(const std::string& /*dir*/) {}

std::unique_ptr<FileLock> MemoryFileSystem::lock(const std::string& /*path*/) {
    return std::make_unique<NoLock>();
}

void MemoryFileSystem::fail_next_sync() {
    std::lock_guard lock(mutex_);
    fail_sync_ = true;
}

void MemoryFileSystem::tear_next_append() {
    std::lock_guard lock(mutex_);
    tear_append_ = true;
}

std::vector<std::uint8_t>& MemoryFileSystem::raw_contents(const std::string& path) {
    std::lock_guard lock(mutex_);
    return *find(path);
}

}  // namespace strata
