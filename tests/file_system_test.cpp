#include "strata/file_system.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using strata::MemoryFileSystem;

namespace {

void write(strata::FileSystem& fs, const std::string& path, const std::string& text,
           bool sync = true) {
    auto f = fs.create(path);
    f->append(text.data(), text.size());
    if (sync) {
        f->sync();
    }
}

std::string read(strata::FileSystem& fs, const std::string& path) {
    const auto bytes = strata::read_whole_file(fs, path);
    return {bytes.begin(), bytes.end()};
}

}  // namespace

TEST(MemoryFileSystem, BasicFileOperations) {
    MemoryFileSystem fs;
    fs.make_dir("/d");
    write(fs, "/d/a", "hello");
    EXPECT_TRUE(fs.exists("/d/a"));
    EXPECT_EQ(read(fs, "/d/a"), "hello");

    auto f = fs.open_append("/d/a");
    f->append(" world", 6);
    f->write_at(0, "J", 1);
    EXPECT_EQ(read(fs, "/d/a"), "Jello world");

    fs.rename("/d/a", "/d/b");
    EXPECT_FALSE(fs.exists("/d/a"));
    EXPECT_EQ(fs.list("/d"), std::vector<std::string>{"b"});
    fs.truncate("/d/b", 5);
    EXPECT_EQ(read(fs, "/d/b"), "Jello");
    fs.remove("/d/b");
    EXPECT_TRUE(fs.list("/d").empty());
    EXPECT_THROW(fs.open_read("/d/b"), std::runtime_error);
}

TEST(MemoryFileSystem, InjectedFaults) {
    MemoryFileSystem fs;
    fs.make_dir("/d");
    auto f = fs.create("/d/a");
    fs.fail_next_sync();
    EXPECT_THROW(f->sync(), std::runtime_error);
    f->sync();  // only one failure

    fs.tear_next_append();
    EXPECT_THROW(f->append("abcdef", 6), std::runtime_error);
    EXPECT_EQ(read(fs, "/d/a"), "abc");
}

TEST(LocalFileSystem, AtomicWriteAndLock) {
    auto& fs = strata::local_file_system();
    const auto dir = (std::filesystem::temp_directory_path() / "strata_fs_test").string();
    std::filesystem::remove_all(dir);
    fs.make_dir(dir);
    const auto path = strata::join_path(dir, "m");
    strata::write_file_atomically(fs, path, "one", 3);
    strata::write_file_atomically(fs, path, "two!", 4);
    EXPECT_EQ(read(fs, path), "two!");
    EXPECT_FALSE(fs.exists(path + ".tmp"));

    {
        const auto lock = fs.lock(strata::join_path(dir, "LOCK"));
        EXPECT_THROW(fs.lock(strata::join_path(dir, "LOCK")), std::runtime_error);
    }
    EXPECT_NO_THROW(fs.lock(strata::join_path(dir, "LOCK")));
    std::filesystem::remove_all(dir);
}

TEST(Paths, JoinAndParent) {
    EXPECT_EQ(strata::join_path("a", "b"), "a/b");
    EXPECT_EQ(strata::join_path("a/", "b"), "a/b");
    EXPECT_EQ(strata::parent_dir("a/b/c"), "a/b");
    EXPECT_EQ(strata::parent_dir("/c"), "/");
    EXPECT_EQ(strata::parent_dir("c"), ".");
}
