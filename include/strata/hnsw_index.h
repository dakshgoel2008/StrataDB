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
#include "strata/quantizer.h"

namespace strata {

class VisitedPool;
class VisitedList;

struct HnswParams {
    std::size_t M = 16;                 // links per node on upper layers; layer 0 allows 2 * M
    std::size_t ef_construction = 200;  // how many candidates to keep while inserting
    std::uint64_t seed = 100;           // makes node levels repeatable
};

// How each vector is kept inside the index.
enum class Encoding : std::uint8_t {
    Float,  // full 4-byte floats
    SQ8,    // one byte per value (ScalarQuantizer)
    PQ,     // one byte per piece (ProductQuantizer)
};

// Full vectors used to re-score candidates found with compressed codes.
// Row i starts at data + i * stride, so a memory-mapped .fvecs file works directly.
struct FullVectors {
    const float* data = nullptr;
    std::size_t stride = 0;
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

    // Same, but each vector is stored as a compressed code instead of full floats.
    // The quantizer must be trained on vectors like the ones that will be added
    // (already normalized, for Cosine).
    HnswIndex(std::size_t dim, Metric metric, std::size_t capacity, HnswParams params,
              ScalarQuantizer quantizer);
    HnswIndex(std::size_t dim, Metric metric, std::size_t capacity, HnswParams params,
              ProductQuantizer quantizer);
    ~HnswIndex();
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    std::size_t dim() const noexcept { return dim_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t capacity() const noexcept { return capacity_; }
    const HnswParams& params() const noexcept { return params_; }
    Encoding encoding() const noexcept { return encoding_; }

    // Bytes one vector takes inside the index (not counting graph links).
    std::size_t bytes_per_vector() const noexcept {
        return encoding_ == Encoding::Float ? dim_ * sizeof(float) : code_size_;
    }

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

    // Search with re-ranking: keeps the best `rerank` candidates (at least k) using the
    // stored codes, re-scores them exactly with `full` vectors, and returns the best k.
    // This recovers most of the accuracy lost to compression.
    void search_rerank(std::span<const float> query, std::size_t k, std::size_t ef_search,
                       std::size_t rerank, const FullVectors& full, std::span<RowId> ids,
                       std::span<float> distances) const;

    // search() for many queries at once. Results for query i are at [i * k, (i + 1) * k).
    void search_batch(std::span<const float> queries, std::size_t k, std::size_t ef_search,
                      std::span<RowId> ids, std::span<float> distances,
                      std::size_t threads = 0) const;

    // Graph statistics. Don't call while add() is running.
    HnswStats stats() const;

    // Simple file format, not crash-safe: a crash during save() can leave a broken file.
    // Only indexes that store full vectors can be saved. Don't call save() while add() is running.
    void save(const std::string& path) const;
    static std::unique_ptr<HnswIndex> load(const std::string& path, std::size_t capacity = 0);

private:
    using NodeId = std::uint32_t;

    struct Candidate {
        float distance;
        NodeId id;
    };

    // Distance from one query to stored nodes, one class per encoding.
    class FloatQuery;
    class Sq8Query;
    class PqQuery;

    HnswIndex(std::size_t dim, Metric metric, std::size_t capacity, HnswParams params,
              Encoding encoding, std::size_t code_size);

    const float* vector_of(NodeId node) const noexcept {
        return vectors_.data() + static_cast<std::size_t>(node) * stride_;
    }
    float* vector_of(NodeId node) noexcept {
        return vectors_.data() + static_cast<std::size_t>(node) * stride_;
    }
    const std::uint8_t* code_of(NodeId node) const noexcept {
        return codes_.data() + static_cast<std::size_t>(node) * code_size_;
    }
    std::uint8_t* code_of(NodeId node) noexcept {
        return codes_.data() + static_cast<std::size_t>(node) * code_size_;
    }

    // Writes a (normalized) vector into the node's slot, compressing it if needed.
    void store(NodeId node, const float* vector);
    // Distance between two stored nodes. Compressed nodes are decoded first.
    float node_distance(NodeId a, NodeId b) const;
    // True if two nodes hold exactly the same stored data.
    bool same_data(NodeId a, NodeId b) const;
    void prefetch_node(NodeId node) const noexcept;

    // Calls fn with the right query-distance object for this index's encoding.
    template <typename Fn>
    decltype(auto) with_query(const float* query, Fn&& fn) const;

    // A link block is [count][slot 0]...[slot max-1].
    NodeId* links(NodeId node, int level) noexcept;
    const NodeId* links(NodeId node, int level) const noexcept;
    std::size_t max_links(int level) const noexcept {
        return level == 0 ? max_links_level0_ : params_.M;
    }

    int random_level(NodeId node) const noexcept;
    void insert(NodeId node, const float* vector);
    template <typename Query>
    void link_node(NodeId node, int level, const Query& distance);

    // Copies a node's neighbor list while holding that node's lock.
    void copy_links(NodeId node, int level, std::vector<NodeId>& out) const;

    // Moves towards the query on one layer until no neighbor is closer.
    template <typename Query>
    void greedy_search(const Query& distance, Candidate& current, int level,
                       std::vector<NodeId>& scratch) const;

    // Best-first search on one layer. Returns up to ef closest nodes, sorted closest first.
    template <typename Query>
    std::vector<Candidate> search_layer(const Query& distance, const std::vector<Candidate>& start,
                                        std::size_t ef, int level, VisitedList& visited) const;

    // Full search from the top layer down. Returns up to ef nodes, closest first.
    template <typename Query>
    std::vector<Candidate> search_graph(const Query& distance, std::size_t ef) const;

    // The query as the index sees it (a normalized copy for Cosine).
    const float* prepare_query(std::span<const float> query, std::vector<float>& buffer) const;

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
    Kernels kernels_;

    Encoding encoding_;
    std::size_t code_size_;  // bytes per compressed vector (unused for Float)
    std::unique_ptr<ScalarQuantizer> sq_;
    std::unique_ptr<ProductQuantizer> pq_;

    AlignedBuffer<float> vectors_;       // used when encoding_ is Float
    AlignedBuffer<std::uint8_t> codes_;  // used otherwise
    std::vector<NodeId> level0_links_;   // capacity blocks of (1 + 2M)
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
