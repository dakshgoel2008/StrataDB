#include "strata/file_system.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace strata {
namespace {

[[noreturn]] void fail(const std::string& what, const std::string& path, int error = errno) {
    throw std::runtime_error(what + " '" + path + "': " + std::strerror(error));
}

// Closes a file descriptor when it goes out of scope.
class Fd {
public:
    Fd(int fd, std::string path) : fd_(fd), path_(std::move(path)) {}
    ~Fd() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;

    int get() const noexcept { return fd_; }
    const std::string& path() const noexcept { return path_; }

private:
    int fd_;
    std::string path_;
};

int open_or_throw(const std::string& path, int flags, mode_t mode = 0644) {
    const int fd = ::open(path.c_str(), flags | O_CLOEXEC, mode);
    if (fd < 0) {
        fail("cannot open", path);
    }
    return fd;
}

void sync_fd(int fd, const std::string& path) {
    if (::fdatasync(fd) != 0) {
        fail("sync failed for", path);
    }
}

std::uint64_t fd_size(int fd, const std::string& path) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        fail("cannot stat", path);
    }
    return static_cast<std::uint64_t>(st.st_size);
}

class LocalWritableFile final : public WritableFile {
public:
    LocalWritableFile(int fd, const std::string& path) : fd_(fd, path), size_(fd_size(fd, path)) {}

    void append(const void* data, std::size_t size) override { write_at(size_, data, size); }

    void write_at(std::uint64_t offset, const void* data, std::size_t size) override {
        const auto* p = static_cast<const char*>(data);
        std::size_t done = 0;
        while (done < size) {
            const ssize_t n =
                ::pwrite(fd_.get(), p + done, size - done, static_cast<off_t>(offset + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                fail("write failed for", fd_.path());
            }
            done += static_cast<std::size_t>(n);
        }
        size_ = std::max(size_, offset + size);
    }

    void sync() override { sync_fd(fd_.get(), fd_.path()); }
    std::uint64_t size() const override { return size_; }

private:
    Fd fd_;
    std::uint64_t size_;
};

class LocalReadableFile final : public ReadableFile {
public:
    LocalReadableFile(int fd, const std::string& path) : fd_(fd, path), size_(fd_size(fd, path)) {}

    std::uint64_t size() const override { return size_; }

    void read(std::uint64_t offset, void* out, std::size_t size) const override {
        auto* p = static_cast<char*>(out);
        std::size_t done = 0;
        while (done < size) {
            const ssize_t n =
                ::pread(fd_.get(), p + done, size - done, static_cast<off_t>(offset + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                fail("read failed for", fd_.path());
            }
            if (n == 0) {
                throw std::runtime_error("'" + fd_.path() + "' is shorter than expected");
            }
            done += static_cast<std::size_t>(n);
        }
    }

private:
    Fd fd_;
    std::uint64_t size_;
};

class LocalFileLock final : public FileLock {
public:
    explicit LocalFileLock(const std::string& path)
        : fd_(open_or_throw(path, O_RDWR | O_CREAT), path) {
        if (::flock(fd_.get(), LOCK_EX | LOCK_NB) != 0) {
            fail("another program is using", path);
        }
    }

private:
    Fd fd_;  // closing it releases the lock, also when the program dies
};

class LocalFileSystem final : public FileSystem {
public:
    std::unique_ptr<WritableFile> create(const std::string& path) override {
        const int fd = open_or_throw(path, O_WRONLY | O_CREAT | O_TRUNC);
        return std::make_unique<LocalWritableFile>(fd, path);
    }

    std::unique_ptr<WritableFile> open_append(const std::string& path) override {
        const int fd = open_or_throw(path, O_WRONLY);
        return std::make_unique<LocalWritableFile>(fd, path);
    }

    std::unique_ptr<ReadableFile> open_read(const std::string& path) override {
        const int fd = open_or_throw(path, O_RDONLY);
        return std::make_unique<LocalReadableFile>(fd, path);
    }

    bool exists(const std::string& path) override {
        std::error_code ec;
        return std::filesystem::exists(path, ec);
    }

    std::vector<std::string> list(const std::string& dir) override {
        std::vector<std::string> names;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            names.push_back(entry.path().filename().string());
        }
        if (ec) {
            fail("cannot list", dir, ec.value());
        }
        return names;
    }

    void make_dir(const std::string& dir) override {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            fail("cannot create directory", dir, ec.value());
        }
    }

    void rename(const std::string& from, const std::string& to) override {
        if (::rename(from.c_str(), to.c_str()) != 0) {
            fail("cannot rename", from);
        }
    }

    void remove(const std::string& path) override {
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            fail("cannot remove", path);
        }
    }

    void truncate(const std::string& path, std::uint64_t size) override {
        const Fd fd(open_or_throw(path, O_WRONLY), path);
        if (::ftruncate(fd.get(), static_cast<off_t>(size)) != 0) {
            fail("cannot truncate", path);
        }
        sync_fd(fd.get(), path);
    }

    void sync_dir(const std::string& dir) override {
        const Fd fd(open_or_throw(dir, O_RDONLY | O_DIRECTORY), dir);
        if (::fsync(fd.get()) != 0) {
            fail("sync failed for directory", dir);
        }
    }

    std::unique_ptr<FileLock> lock(const std::string& path) override {
        return std::make_unique<LocalFileLock>(path);
    }
};

}  // namespace

FileSystem& local_file_system() {
    static LocalFileSystem fs;
    return fs;
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    return dir.back() == '/' ? dir + name : dir + "/" + name;
}

std::string parent_dir(const std::string& path) {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return ".";
    }
    return slash == 0 ? "/" : path.substr(0, slash);
}

std::vector<std::uint8_t> read_whole_file(FileSystem& fs, const std::string& path) {
    const auto file = fs.open_read(path);
    std::vector<std::uint8_t> data(file->size());
    file->read(0, data.data(), data.size());
    return data;
}

void write_file_atomically(FileSystem& fs, const std::string& path, const void* data,
                           std::size_t size) {
    const std::string temp = path + ".tmp";
    {
        const auto file = fs.create(temp);
        file->append(data, size);
        file->sync();
    }
    fs.rename(temp, path);
    fs.sync_dir(parent_dir(path));
}

}  // namespace strata
