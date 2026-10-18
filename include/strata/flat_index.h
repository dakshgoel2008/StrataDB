#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "strata/aligned_buffer.h"
#include "strata/metric.h"

namespace strata {

// Row number of a vector inside an index. -1 means "no result".
using RowId = std::int64_t;
inline constexpr RowId kNoRow = -1;

// Exact search: compares the query with every stored vector.
// Slow on big data, but always correct, so it's used to check the faster indexes.
class FlatIndex {
public:
    FlatIndex(std::size_t dim, Metric metric);

    std::size_t dim() const noexcept { return dim_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return count_; }

    // Makes room for n vectors in total, so adding them later doesn't reallocate.
    void reserve(std::size_t n) { vectors_.reserve(n * stride_); }

    // Adds vectors stored back to back (n * dim floats).
    // They get row ids size(), size() + 1, ... in order.
    void add(std::span<const float> vectors);

    // Finds the k closest vectors to one query, closest first.
    // ids and distances must hold k items; unused slots get kNoRow and +infinity.
    void search(std::span<const float> query, std::size_t k, std::span<RowId> ids,
                std::span<float> distances) const;

    // Same as search() for many queries, split across threads (0 = all cores).
    // Results for query i are at [i * k, (i + 1) * k).
    void search_batch(std::span<const float> queries, std::size_t k, std::span<RowId> ids,
                      std::span<float> distances, std::size_t threads = 0) const;

private:
    const float* row(std::size_t i) const noexcept { return vectors_.data() + i * stride_; }

    std::size_t dim_;
    std::size_t stride_;  // dim rounded up so every row starts on a 64-byte boundary
    Metric metric_;
    std::size_t count_ = 0;
    AlignedBuffer<float> vectors_;
};

}  // namespace strata
