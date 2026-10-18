#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bench/common/vecs_io.h"

namespace strata::bench {

// Share of the true top-k neighbors found in our top-k results, averaged over queries.
// Both tables have one row per query; -1 in results means "no result".
double recall_at_k(const Matrix<std::int32_t>& results, const Matrix<std::int32_t>& groundtruth,
                   std::size_t k);

struct LatencySummary {
    std::size_t count = 0;
    double mean_us = 0;
    double p50_us = 0;
    double p95_us = 0;
    double p99_us = 0;
    double p999_us = 0;
    double max_us = 0;
};

// Collects how long each query took and reports the slow cases (p99 etc).
class LatencyRecorder {
public:
    void reserve(std::size_t n) { samples_ns_.reserve(n); }
    void add(std::uint64_t nanos) { samples_ns_.push_back(nanos); }
    std::size_t count() const { return samples_ns_.size(); }
    void clear() { samples_ns_.clear(); }

    // Throws if nothing was recorded.
    LatencySummary summarize() const;

private:
    std::vector<std::uint64_t> samples_ns_;
};

// Smallest value that at least p% of the samples are <= to. Input must be sorted.
std::uint64_t percentile_sorted(const std::vector<std::uint64_t>& sorted, double p);

}  // namespace strata::bench
