#include "strata/crc32c.h"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

using strata::crc32c;
using strata::crc32c_portable;

TEST(Crc32c, KnownValues) {
    // Check values from RFC 3720 (iSCSI), section B.4.
    const std::string digits = "123456789";
    EXPECT_EQ(crc32c(digits.data(), digits.size()), 0xE3069283u);
    EXPECT_EQ(crc32c_portable(digits.data(), digits.size()), 0xE3069283u);

    std::vector<unsigned char> zeros(32, 0);
    std::vector<unsigned char> ones(32, 0xff);
    EXPECT_EQ(crc32c(zeros.data(), zeros.size()), 0x8A9136AAu);
    EXPECT_EQ(crc32c(ones.data(), ones.size()), 0x62A8AB43u);
    EXPECT_EQ(crc32c(nullptr, 0), 0u);
}

TEST(Crc32c, BothVersionsAgreeOnEveryLengthAndOffset) {
    std::mt19937 rng(1);
    std::vector<unsigned char> data(300);
    for (auto& b : data) {
        b = static_cast<unsigned char>(rng());
    }
    for (std::size_t offset = 0; offset < 9; ++offset) {
        for (std::size_t len = 0; len + offset <= data.size(); len += 7) {
            ASSERT_EQ(crc32c(data.data() + offset, len), crc32c_portable(data.data() + offset, len))
                << "offset " << offset << " length " << len;
        }
    }
}

TEST(Crc32c, PiecesGiveTheSameResultAsTheWhole) {
    const std::string text = "the quick brown fox jumps over the lazy dog";
    const std::uint32_t whole = crc32c(text.data(), text.size());
    for (std::size_t cut = 0; cut <= text.size(); ++cut) {
        const std::uint32_t first = crc32c(text.data(), cut);
        EXPECT_EQ(crc32c(text.data() + cut, text.size() - cut, first), whole);
    }
}

TEST(Crc32c, DetectsEverySingleBitFlip) {
    std::vector<unsigned char> data(64, 0x5a);
    const std::uint32_t good = crc32c(data.data(), data.size());
    for (std::size_t bit = 0; bit < data.size() * 8; ++bit) {
        data[bit / 8] ^= static_cast<unsigned char>(1u << (bit % 8));
        EXPECT_NE(crc32c(data.data(), data.size()), good);
        data[bit / 8] ^= static_cast<unsigned char>(1u << (bit % 8));
    }
}
