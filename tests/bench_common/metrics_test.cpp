#include "bench/common/metrics.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sb = strata::bench;
using IdMatrix = sb::Matrix<std::int32_t>;

TEST(RecallAtK, PerfectResults) {
    IdMatrix gt{2, 3, {1, 2, 3, 7, 8, 9}};
    EXPECT_DOUBLE_EQ(sb::recall_at_k(gt, gt, 3), 1.0);
}

TEST(RecallAtK, OrderInsideTopKDoesNotMatter) {
    IdMatrix gt{1, 3, {1, 2, 3}};
    IdMatrix res{1, 3, {3, 1, 2}};
    EXPECT_DOUBLE_EQ(sb::recall_at_k(res, gt, 3), 1.0);
}

TEST(RecallAtK, PartialMatches) {
    IdMatrix gt{2, 2, {1, 2, 5, 6}};
    IdMatrix res{2, 2, {1, 9, 9, 9}};  // query 0: 1/2 hits, query 1: 0/2
    EXPECT_DOUBLE_EQ(sb::recall_at_k(res, gt, 2), 0.25);
}

TEST(RecallAtK, OnlyFirstKColumnsCount) {
    // recall@2 must only look at the first 2 true ids (1 and 2).
    IdMatrix gt{1, 4, {1, 2, 3, 4}};
    IdMatrix res{1, 2, {3, 4}};
    EXPECT_DOUBLE_EQ(sb::recall_at_k(res, gt, 2), 0.0);
}

TEST(RecallAtK, PaddingNeverMatches) {
    IdMatrix gt{1, 2, {1, 2}};
    IdMatrix res{1, 2, {1, -1}};
    EXPECT_DOUBLE_EQ(sb::recall_at_k(res, gt, 2), 0.5);
}

TEST(RecallAtK, RejectsBadShapes) {
    IdMatrix gt{1, 2, {1, 2}};
    IdMatrix res{1, 2, {1, 2}};
    IdMatrix two_queries{2, 2, {1, 2, 1, 2}};
    EXPECT_THROW(sb::recall_at_k(res, gt, 0), std::invalid_argument);
    EXPECT_THROW(sb::recall_at_k(res, gt, 3), std::invalid_argument);
    EXPECT_THROW(sb::recall_at_k(two_queries, gt, 2), std::invalid_argument);
}

TEST(Percentile, NearestRank) {
    const std::vector<std::uint64_t> v = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    EXPECT_EQ(sb::percentile_sorted(v, 0), 10u);   // smallest position is 1
    EXPECT_EQ(sb::percentile_sorted(v, 50), 50u);  // 5th value
    EXPECT_EQ(sb::percentile_sorted(v, 51), 60u);  // rounds up to the 6th value
    EXPECT_EQ(sb::percentile_sorted(v, 99), 100u);
    EXPECT_EQ(sb::percentile_sorted(v, 100), 100u);
}

TEST(Percentile, RejectsBadInput) {
    EXPECT_THROW(sb::percentile_sorted({}, 50), std::logic_error);
    EXPECT_THROW(sb::percentile_sorted({1}, 101), std::invalid_argument);
}

TEST(LatencyRecorder, SummaryInMicroseconds) {
    sb::LatencyRecorder rec;
    for (std::uint64_t i = 1; i <= 1000; ++i) {
        rec.add(i * 1000);  // 1us .. 1000us, added in order
    }
    const auto s = rec.summarize();
    EXPECT_EQ(s.count, 1000u);
    EXPECT_DOUBLE_EQ(s.mean_us, 500.5);
    EXPECT_DOUBLE_EQ(s.p50_us, 500.0);
    EXPECT_DOUBLE_EQ(s.p99_us, 990.0);
    EXPECT_DOUBLE_EQ(s.p999_us, 999.0);
    EXPECT_DOUBLE_EQ(s.max_us, 1000.0);
}

TEST(LatencyRecorder, UnsortedInputIsHandled) {
    sb::LatencyRecorder rec;
    for (std::uint64_t v : {5000u, 1000u, 3000u}) {
        rec.add(v);
    }
    EXPECT_DOUBLE_EQ(rec.summarize().p50_us, 3.0);
}

TEST(LatencyRecorder, EmptyThrows) {
    sb::LatencyRecorder rec;
    EXPECT_THROW(rec.summarize(), std::logic_error);
}
