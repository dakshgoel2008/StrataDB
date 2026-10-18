#include "bench/common/vecs_io.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb = strata::bench;
namespace fs = std::filesystem;

namespace {

// Gives every test its own empty directory and removes it afterwards.
class VecsIoTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = fs::temp_directory_path() / (std::string("strata_vecs_io_") + info->name());
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    std::string path(const std::string& name) const { return (dir_ / name).string(); }

    // Writes raw int32 values so tests can build broken files.
    void write_raw(const std::string& p, const std::vector<std::int32_t>& words) const {
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char*>(words.data()),
                  static_cast<std::streamsize>(words.size() * sizeof(std::int32_t)));
    }

    fs::path dir_;
};

sb::Matrix<float> make_float_matrix(std::size_t rows, std::size_t dim) {
    sb::Matrix<float> m{rows, dim, std::vector<float>(rows * dim)};
    for (std::size_t i = 0; i < m.data.size(); ++i) {
        m.data[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }
    return m;
}

}  // namespace

TEST_F(VecsIoTest, FvecsRoundTrip) {
    const auto m = make_float_matrix(5, 3);
    sb::write_fvecs(path("a.fvecs"), m);

    const auto back = sb::read_fvecs(path("a.fvecs"));
    EXPECT_EQ(back.rows, 5u);
    EXPECT_EQ(back.dim, 3u);
    EXPECT_EQ(back.data, m.data);
}

TEST_F(VecsIoTest, IvecsRoundTrip) {
    sb::Matrix<std::int32_t> m{2, 4, {1, -2, 3, 4, 5, 6, 7, 2147483647}};
    sb::write_ivecs(path("a.ivecs"), m);

    const auto back = sb::read_ivecs(path("a.ivecs"));
    EXPECT_EQ(back.rows, 2u);
    EXPECT_EQ(back.dim, 4u);
    EXPECT_EQ(back.data, m.data);
}

TEST_F(VecsIoTest, InspectReportsShapeWithoutReading) {
    sb::write_fvecs(path("a.fvecs"), make_float_matrix(7, 16));
    const auto shape = sb::inspect_vecs(path("a.fvecs"), sizeof(float));
    EXPECT_EQ(shape.rows, 7u);
    EXPECT_EQ(shape.dim, 16u);
}

TEST_F(VecsIoTest, MaxRowsReadsPrefix) {
    const auto m = make_float_matrix(10, 2);
    sb::write_fvecs(path("a.fvecs"), m);

    const auto part = sb::read_fvecs(path("a.fvecs"), 3);
    ASSERT_EQ(part.rows, 3u);
    EXPECT_EQ(part.data, std::vector<float>(m.data.begin(), m.data.begin() + 6));

    // Asking for more rows than exist returns all of them.
    EXPECT_EQ(sb::read_fvecs(path("a.fvecs"), 100).rows, 10u);
}

TEST_F(VecsIoTest, FirstRowReadsFromTheMiddle) {
    const auto m = make_float_matrix(10, 2);
    sb::write_fvecs(path("a.fvecs"), m);

    const auto part = sb::read_fvecs(path("a.fvecs"), 3, 4);  // rows 4, 5, 6
    ASSERT_EQ(part.rows, 3u);
    EXPECT_EQ(part.data, std::vector<float>(m.data.begin() + 8, m.data.begin() + 14));

    // Near the end only the remaining rows come back; past the end, none.
    EXPECT_EQ(sb::read_fvecs(path("a.fvecs"), 5, 8).rows, 2u);
    EXPECT_EQ(sb::read_fvecs(path("a.fvecs"), 0, 7).rows, 3u);
    EXPECT_EQ(sb::read_fvecs(path("a.fvecs"), 0, 20).rows, 0u);
}

TEST_F(VecsIoTest, BvecsAreWidenedToFloat) {
    // One row, dim=4, bytes {0, 1, 200, 255}.
    const std::string p = path("a.bvecs");
    {
        std::ofstream out(p, std::ios::binary);
        const std::int32_t dim = 4;
        const unsigned char bytes[] = {0, 1, 200, 255};
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    const auto m = sb::read_bvecs(p);
    ASSERT_EQ(m.rows, 1u);
    EXPECT_EQ(m.data, (std::vector<float>{0.f, 1.f, 200.f, 255.f}));
}

TEST_F(VecsIoTest, EmptyFileHasZeroRows) {
    write_raw(path("empty.fvecs"), {});
    const auto m = sb::read_fvecs(path("empty.fvecs"));
    EXPECT_EQ(m.rows, 0u);
    EXPECT_TRUE(m.data.empty());
}

TEST_F(VecsIoTest, MissingFileThrows) {
    EXPECT_THROW(sb::read_fvecs(path("nope.fvecs")), std::runtime_error);
}

TEST_F(VecsIoTest, TruncatedFileThrows) {
    // dim=3 but only 2 values follow.
    write_raw(path("bad.fvecs"), {3, 0, 0});
    EXPECT_THROW(sb::read_fvecs(path("bad.fvecs")), std::runtime_error);
}

TEST_F(VecsIoTest, NonPositiveDimThrows) {
    write_raw(path("zero.fvecs"), {0, 0});
    EXPECT_THROW(sb::read_fvecs(path("zero.fvecs")), std::runtime_error);
    write_raw(path("neg.fvecs"), {-1, 0});
    EXPECT_THROW(sb::read_fvecs(path("neg.fvecs")), std::runtime_error);
}

TEST_F(VecsIoTest, InconsistentRowDimThrows) {
    // Row 1 claims dim=1 but the file size still looks valid,
    // so only the per-row check can catch it.
    write_raw(path("mixed.fvecs"), {2, 0, 0, 1, 0, 0});
    EXPECT_THROW(sb::read_fvecs(path("mixed.fvecs")), std::runtime_error);
}

TEST_F(VecsIoTest, WriteRejectsBadShape) {
    sb::Matrix<float> m{2, 3, std::vector<float>(5)};  // needs 6 values
    EXPECT_THROW(sb::write_fvecs(path("x.fvecs"), m), std::runtime_error);
}
