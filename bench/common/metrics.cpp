#include "bench/common/metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::bench {

double recall_at_k(const Matrix<std::int32_t>& results, const Matrix<std::int32_t>& groundtruth,
                   std::size_t k) {
    if (k == 0) {
        throw std::invalid_argument("k must be > 0");
    }
    if (results.rows != groundtruth.rows) {
        throw std::invalid_argument("results and groundtruth have different query counts");
    }
    if (results.dim < k || groundtruth.dim < k) {
        throw std::invalid_argument("results or groundtruth has fewer than k columns");
    }
    if (results.rows == 0) {
        return 0.0;
    }

    std::vector<std::int32_t> truth(k);
    std::size_t hits = 0;
    for (std::size_t q = 0; q < results.rows; ++q) {
        // Sort the true ids once so each lookup is a fast binary search.
        const std::int32_t* gt = groundtruth.row(q);
        std::copy(gt, gt + k, truth.begin());
        std::sort(truth.begin(), truth.end());

        const std::int32_t* res = results.row(q);
        for (std::size_t j = 0; j < k; ++j) {
            if (std::binary_search(truth.begin(), truth.end(), res[j])) {
                ++hits;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(results.rows * k);
}

std::uint64_t percentile_sorted(const std::vector<std::uint64_t>& sorted, double p) {
    if (sorted.empty()) {
        throw std::logic_error("percentile of empty sample");
    }
    if (p < 0.0 || p > 100.0) {
        throw std::invalid_argument("percentile must be in [0, 100]");
    }
    // Position is ceil(p% of n). The tiny epsilon stops float rounding
    // from pushing whole numbers up (99.9/100*1000 gives 999.0000000000001).
    constexpr double kEps = 1e-9;
    const auto n = static_cast<double>(sorted.size());
    auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * n - kEps));
    rank = std::clamp<std::size_t>(rank, 1, sorted.size());
    return sorted[rank - 1];
}

LatencySummary LatencyRecorder::summarize() const {
    if (samples_ns_.empty()) {
        throw std::logic_error("no latency samples recorded");
    }
    std::vector<std::uint64_t> sorted = samples_ns_;
    std::sort(sorted.begin(), sorted.end());

    const auto to_us = [](std::uint64_t ns) { return static_cast<double>(ns) / 1000.0; };
    // Start from a uint64 zero; a plain 0 would add up in int and overflow.
    const std::uint64_t total_ns = std::accumulate(sorted.begin(), sorted.end(), std::uint64_t{0});

    LatencySummary s;
    s.count = sorted.size();
    s.mean_us = to_us(total_ns) / static_cast<double>(sorted.size());
    s.p50_us = to_us(percentile_sorted(sorted, 50.0));
    s.p95_us = to_us(percentile_sorted(sorted, 95.0));
    s.p99_us = to_us(percentile_sorted(sorted, 99.0));
    s.p999_us = to_us(percentile_sorted(sorted, 99.9));
    s.max_us = to_us(sorted.back());
    return s;
}

}  // namespace strata::bench
