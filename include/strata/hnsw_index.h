#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "strata/aligned_buffer.h"
#include "strata/distance.h"
#include "strata/flat_index.h"
#include "strata/metric.h"

namespace strata {

class VisitedPool;
class VisitedList;

struct HnswParams {
    std::size_t M = 16;                 // links per node on upper layers; layer 0 allows 2 * M
    std::size_t ef_construction = 200;  // how many candidates to keep while inserting
    std::uint64_t seed = 100;           // makes node levels repeatable
};

struct HnswStats {
    std::size_t size = 0;
    int max_level = -1;
    std::vector<std::size_t> nodes_per_level;  // nodes_per_level[l] = nodes that reach level l
    double avg_links_level0 = 0;
    std::size_t max_links_level0 = 0;
    std::size_t memory_bytes = 0;
};

// Approximate nearest neighbor search with an HNSW graph
// (Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
// using Hierarchical Navigable Small World graphs").
//
// The graph has layers. Every vector is on layer 0; fewer and fewer reach the
// higher layers. A search starts at the top, moves greedily towards the query,
// and drops one layer at a time until it reaches layer 0.
//
// add() and search() can be called from many threads at the same time.
class HnswIndex {
public:
    // capacity is the most vectors this index can ever hold. Memory for them is
    // set aside now, so it never moves while other threads are reading.
    HnswIndex(std::size_t dim, Metric metric, std::size_t capacity, HnswParams params = {});
    ~HnswIndex();
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    std::size_t dim() const noexcept { return dim_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t capacity() const noexcept { return capacity_; }
    const HnswParams& params() const noexcept { return params_; }

    // Number of vectors added, including ones still being inserted by other threads.
    std::size_t size() const noexcept { return count_.load(std::memory_order_acquire); }

    // Adds vectors stored back to back (n * dim floats) using `threads` threads
    // (0 = all cores). Vector i gets row id size() + i. Throws if capacity would be exceeded.
    void add(std::span<const float> vectors, std::size_t threads = 1);

    // Finds about the k closest vectors, closest first. ef_search is how many
    // candidates to track: higher is more accurate but slower (at least k is used).
    // Unused output slots get kNoRow and +infinity.
    void search(std::span<const float> query, std::size_t k, std::size_t ef_search,
                std::span<RowId> ids, std::span<float> distances) const;

    // search() for many queries at once. Results for query i are at [i * k, (i + 1) * k).
    void search_batch(std::span<const float> queries, std::size_t k, std::size_t ef_search,
                      std::span<RowId> ids, std::span<float> distances,
                      std::size_t threads = 0) const;

    // Graph statistics. Don't call while add() is running.
    HnswStats stats() const;

    // Simple file format, not crash-safe: a crash during save() can leave a broken file.
    // Don't call save() while add() is running.
    void save(const std::string& path) const;
    static std::unique_ptr<HnswIndex> load(const std::string& path, std::size_t capacity = 0);

private:
    using NodeId = std::uint32_t;

    struct Candidate {
        float distance;
        NodeId id;
    };

    const float* vector_of(NodeId node) const noexcept {
        return vectors_.data() + static_cast<std::size_t>(node) * stride_;
    }
    float* vector_of(NodeId node) noexcept {
        return vectors_.data() + static_cast<std::size_t>(node) * stride_;
    }

    // A link block is [count][slot 0]...[slot max-1].
    NodeId* links(NodeId node, int level) noexcept;
    const NodeId* links(NodeId node, int level) const noexcept;
    std::size_t max_links(int level) const noexcept {
        return level == 0 ? max_links_level0_ : params_.M;
    }

    int random_level(NodeId node) const noexcept;
    void insert(NodeId node, const float* vector);

    // Copies a node's neighbor list while holding that node's lock.
    void copy_links(NodeId node, int level, std::vector<NodeId>& out) const;

    // Moves towards the query on one layer until no neighbor is closer.
    void greedy_search(const float* query, Candidate& current, int level,
                       std::vector<NodeId>& scratch) const;

    // Best-first search on one layer. Returns up to ef closest nodes, sorted closest first.
    std::vector<Candidate> search_layer(const float* query, const std::vector<Candidate>& start,
                                        std::size_t ef, int level, VisitedList& visited) const;

    // Keeps at most m candidates, preferring ones that point in different directions
    // (the paper's heuristic). Input must be sorted closest first.
    void select_neighbors(std::vector<Candidate>& candidates, std::size_t m) const;

    // Adds a link from `node` to `new_node`, pruning node's list if it is full.
    void connect(NodeId node, NodeId new_node, float distance, int level);

    std::size_t dim_;
    std::size_t stride_;
    Metric metric_;
    std::size_t capacity_;
    HnswParams params_;
    std::size_t max_links_level0_;
    double level_factor_;
    DistanceFn distance_;

    AlignedBuffer<float> vectors_;
    std::vector<NodeId> level0_links_;  // capacity blocks of (1 + 2M)
    std::vector<std::vector<NodeId>>
        upper_links_;  // per node: one block of (1 + M) per level above 0
    std::vector<std::uint8_t> levels_;
    std::unique_ptr<std::mutex[]> node_locks_;

    // Entry point and top level packed together so readers always see a matching pair.
    std::mutex entry_lock_;
    std::atomic<std::uint64_t> entry_;
    std::atomic<std::size_t> count_{0};

    std::unique_ptr<VisitedPool> visited_pool_;
};

}  // namespace strata
