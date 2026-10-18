#include "strata/aligned_buffer.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

bool is_aligned(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % strata::AlignedBuffer<float>::kAlignment == 0;
}

}  // namespace

TEST(AlignedBuffer, StartsEmpty) {
    strata::AlignedBuffer<float> buf;
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(AlignedBuffer, ResizeZeroFillsAndAligns) {
    strata::AlignedBuffer<float> buf;
    buf.resize(5);
    ASSERT_EQ(buf.size(), 5u);
    EXPECT_TRUE(is_aligned(buf.data()));
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_EQ(buf.data()[i], 0.0f);
    }
}

TEST(AlignedBuffer, GrowingKeepsContentsAndAlignment) {
    strata::AlignedBuffer<float> buf;
    for (std::size_t n = 1; n <= 1000; ++n) {
        buf.resize(n);
        buf.data()[n - 1] = static_cast<float>(n);
        ASSERT_TRUE(is_aligned(buf.data()));
    }
    for (std::size_t i = 0; i < 1000; ++i) {
        ASSERT_EQ(buf.data()[i], static_cast<float>(i + 1));
    }
}

TEST(AlignedBuffer, ReserveDoesNotChangeSize) {
    strata::AlignedBuffer<float> buf;
    buf.resize(3);
    buf.data()[2] = 7.0f;
    buf.reserve(100);
    EXPECT_EQ(buf.size(), 3u);
    EXPECT_GE(buf.capacity(), 100u);
    EXPECT_EQ(buf.data()[2], 7.0f);
}

TEST(AlignedBuffer, ShrinkThenGrowZeroFillsAgain) {
    strata::AlignedBuffer<float> buf;
    buf.resize(4);
    buf.data()[3] = 9.0f;
    buf.resize(2);
    buf.resize(4);
    EXPECT_EQ(buf.data()[3], 0.0f);
}
