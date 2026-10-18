#include "strata/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "strata/parallel.h"
#include "visited_list.h"

namespace strata {
namespace {

constexpr std::uint64_t kNoEntry = ~std::uint64_t{0};
constexpr int kMaxLevel = 30;
constexpr std::size_t kFloatsPerCacheLine = 16;

std::uint64_t pack_entry(std::uint32_t node, int level) {
    return (static_cast<std::uint64_t>(level) << 32) | node;
}

std::uint32_t entry_node(std::uint64_t entry) { return static_cast<std::uint32_t>(entry); }

int entry_level(std::uint64_t entry) { return static_cast<int>(entry >> 32); }

// Turns a number into a well-mixed random-looking 64-bit value (splitmix64).
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// True if a is closer than b. Equal distances are ordered by id so results repeat exactly.
template <typename C>
bool closer(const C& a, const C& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

template <typename C>
bool farther(const C& a, const C& b) {
    return closer(b, a);
}

// Small helpers for the temporary file format.
struct FileCloser {
    void operator()(std::FILE* f) const noexcept { std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

void write_bytes(std::FILE* f, const void* data, std::size_t bytes) {
    if (bytes > 0 && std::fwrite(data, 1, bytes, f) != bytes) {
        throw std::runtime_error("write failed while saving HNSW index");
    }
}

void read_bytes(std::FILE* f, void* data, std::size_t bytes) {
    if (bytes > 0 && std::fread(data, 1, bytes, f) != bytes) {
        throw std::runtime_error("HNSW file is cut off or unreadable");
    }
}

template <typename T>
void write_value(std::FILE* f, T value) {
    write_bytes(f, &value, sizeof(value));
}

template <typename T>
T read_value(std::FILE* f) {
    T value{};
    read_bytes(f, &value, sizeof(value));
    return value;
}

constexpr char kMagic[8] = {'S', 'T', 'R', 'H', 'N', 'S', 'W', '1'};
constexpr std::uint32_t kFileVersion = 1;

}  // namespace

HnswIndex::HnswIndex(std::size_t dim, Metric metric, std::size_t capacity, HnswParams params)
    : dim_(dim),
      stride_((dim + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine),
      metric_(metric),
      capacity_(capacity),
      params_(params),
      max_links_level0_(2 * params.M),
      level_factor_(1.0 / std::log(static_cast<double>(params.M))),
      distance_(distance_function(metric)),
      entry_(kNoEntry) {
    if (dim == 0) {
        throw std::invalid_argument("dim must be > 0");
    }
    if (params.M < 2) {
        throw std::invalid_argument("M must be at least 2");
    }
    if (capacity == 0 || capacity >= std::numeric_limits<NodeId>::max()) {
        throw std::invalid_argument("capacity must be between 1 and 2^32 - 2");
    }
    vectors_.resize(capacity * stride_);
    level0_links_.assign(capacity * (1 + max_links_level0_), 0);
    upper_links_.resize(capacity);
    levels_.assign(capacity, 0);
    node_locks_ = std::make_unique<std::mutex[]>(capacity);
    visited_pool_ = std::make_unique<VisitedPool>(capacity);
}

HnswIndex::~HnswIndex() = default;

HnswIndex::NodeId* HnswIndex::links(NodeId node, int level) noexcept {
    if (level == 0) {
        return level0_links_.data() + static_cast<std::size_t>(node) * (1 + max_links_level0_);
    }
    return upper_links_[node].data() + static_cast<std::size_t>(level - 1) * (1 + params_.M);
}

const HnswIndex::NodeId* HnswIndex::links(NodeId node, int level) const noexcept {
    return const_cast<HnswIndex*>(this)->links(node, level);
}

// Level = floor(-ln(u) / ln(M)) with u uniform in (0, 1]. So 1 in M nodes reaches
// level 1, 1 in M^2 reaches level 2, and so on. u comes from the node id, so a
// node always gets the same level no matter which thread inserts it.
int HnswIndex::random_level(NodeId node) const noexcept {
    const std::uint64_t r = mix(params_.seed ^ mix(node));
    const double u = (static_cast<double>(r >> 11) + 1.0) * 0x1.0p-53;
    const double level = -std::log(u) * level_factor_;
    return std::min(static_cast<int>(level), kMaxLevel);
}

void HnswIndex::add(std::span<const float> vectors, std::size_t threads) {
    if (vectors.size() % dim_ != 0) {
        throw std::invalid_argument("vector data size is not a multiple of dim");
    }
    const std::size_t n = vectors.size() / dim_;
    if (n == 0) {
        return;
    }

    // Reserve a block of ids up front, so vector i always gets id first + i.
    std::size_t first = count_.load();
    do {
        if (first + n > capacity_) {
            throw std::length_error("HNSW index is full");
        }
    } while (!count_.compare_exchange_weak(first, first + n));

    parallel_for(n, threads, [&](std::size_t i) {
        insert(static_cast<NodeId>(first + i), vectors.data() + i * dim_);
    });
}

void HnswIndex::insert(NodeId node, const float* vector) {
    float* own = vector_of(node);
    std::copy_n(vector, dim_, own);
    if (metric_ == Metric::Cosine) {
        normalize(own, dim_);
    }

    const int level = random_level(node);
    levels_[node] = static_cast<std::uint8_t>(level);
    if (level > 0) {
        upper_links_[node].assign(static_cast<std::size_t>(level) * (1 + params_.M), 0);
    }

    // Only a node that becomes the new top keeps this lock for its whole insert.
    std::unique_lock top_lock(entry_lock_);
    const std::uint64_t entry = entry_.load(std::memory_order_acquire);
    if (entry == kNoEntry) {
        entry_.store(pack_entry(node, level), std::memory_order_release);
        return;
    }
    const int top_level = entry_level(entry);
    if (level <= top_level) {
        top_lock.unlock();
    }

    // Walk down the layers above this node's level, one greedy step at a time.
    std::vector<NodeId> scratch;
    Candidate current{distance_(own, vector_of(entry_node(entry)), dim_), entry_node(entry)};
    for (int l = top_level; l > level; --l) {
        greedy_search(own, current, l, scratch);
    }

    // On each of this node's layers, find close nodes and link to them both ways.
    VisitedHandle visited(*visited_pool_);
    std::vector<Candidate> start{current};
    for (int l = std::min(level, top_level); l >= 0; --l) {
        std::vector<Candidate> found =
            search_layer(own, start, params_.ef_construction, l, *visited);
        // Another thread may already have linked to this node, so it can show up here.
        std::erase_if(found, [node](const Candidate& c) { return c.id == node; });

        std::vector<Candidate> neighbors = found;
        select_neighbors(neighbors, params_.M);
        {
            // Keep any links other threads added in the meantime.
            std::lock_guard lock(node_locks_[node]);
            NodeId* block = links(node, l);
            for (NodeId i = 0; i < block[0]; ++i) {
                const NodeId other = block[1 + i];
                if (std::none_of(neighbors.begin(), neighbors.end(),
                                 [other](const Candidate& c) { return c.id == other; })) {
                    neighbors.push_back({distance_(own, vector_of(other), dim_), other});
                }
            }
            if (neighbors.size() > max_links(l)) {
                std::sort(neighbors.begin(), neighbors.end(), closer<Candidate>);
                select_neighbors(neighbors, max_links(l));
            }
            block[0] = static_cast<NodeId>(neighbors.size());
            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                block[1 + i] = neighbors[i].id;
            }
        }
        for (const Candidate& n : neighbors) {
            connect(n.id, node, n.distance, l);
        }
        start = std::move(found);
    }

    if (level > top_level) {
        entry_.store(pack_entry(node, level), std::memory_order_release);
    }
}

void HnswIndex::copy_links(NodeId node, int level, std::vector<NodeId>& out) const {
    std::lock_guard lock(node_locks_[node]);
    const NodeId* block = links(node, level);
    out.assign(block + 1, block + 1 + block[0]);
}

void HnswIndex::greedy_search(const float* query, Candidate& current, int level,
                              std::vector<NodeId>& scratch) const {
    bool moved = true;
    while (moved) {
        moved = false;
        copy_links(current.id, level, scratch);
        for (NodeId n : scratch) {
            const float d = distance_(query, vector_of(n), dim_);
            if (d < current.distance) {
                current = {d, n};
                moved = true;
            }
        }
    }
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const float* query,
                                                          const std::vector<Candidate>& start,
                                                          std::size_t ef, int level,
                                                          VisitedList& visited) const {
    visited.start_new_search();
    std::vector<Candidate> to_visit;  // closest on top
    std::vector<Candidate> best;      // farthest on top, at most ef items
    for (const Candidate& c : start) {
        if (!visited.check_and_mark(c.id)) {
            to_visit.push_back(c);
            std::push_heap(to_visit.begin(), to_visit.end(), farther<Candidate>);
            best.push_back(c);
            std::push_heap(best.begin(), best.end(), closer<Candidate>);
        }
    }
    while (best.size() > ef) {
        std::pop_heap(best.begin(), best.end(), closer<Candidate>);
        best.pop_back();
    }

    std::vector<NodeId> neighbors;
    while (!to_visit.empty()) {
        const Candidate current = to_visit.front();
        // Stop once the closest unexplored node is farther than everything we keep.
        if (best.size() >= ef && current.distance > best.front().distance) {
            break;
        }
        std::pop_heap(to_visit.begin(), to_visit.end(), farther<Candidate>);
        to_visit.pop_back();

        copy_links(current.id, level, neighbors);
        for (NodeId n : neighbors) {
            if (visited.check_and_mark(n)) {
                continue;
            }
            const float d = distance_(query, vector_of(n), dim_);
            if (best.size() < ef || d < best.front().distance) {
                to_visit.push_back({d, n});
                std::push_heap(to_visit.begin(), to_visit.end(), farther<Candidate>);
                best.push_back({d, n});
                std::push_heap(best.begin(), best.end(), closer<Candidate>);
                if (best.size() > ef) {
                    std::pop_heap(best.begin(), best.end(), closer<Candidate>);
                    best.pop_back();
                }
            }
        }
    }
    std::sort_heap(best.begin(), best.end(), closer<Candidate>);
    return best;
}

// A candidate is skipped if it is closer to an already chosen neighbor than to
// the base node. This spreads links in different directions instead of
// bunching them in one cluster, which keeps the graph well connected.
//
// Exact copies of a chosen neighbor would never be skipped by that rule and
// could fill the whole list, trapping searches. They only get leftover slots.
void HnswIndex::select_neighbors(std::vector<Candidate>& candidates, std::size_t m) const {
    if (candidates.size() <= m) {
        return;
    }
    std::vector<Candidate> chosen;
    std::vector<Candidate> copies;
    chosen.reserve(m);
    for (const Candidate& c : candidates) {
        const float* v = vector_of(c.id);
        bool skip = false;
        bool is_copy = false;
        for (const Candidate& kept : chosen) {
            const float* w = vector_of(kept.id);
            if (std::equal(v, v + dim_, w)) {
                is_copy = true;
                break;
            }
            if (distance_(v, w, dim_) < c.distance) {
                skip = true;
                break;
            }
        }
        if (is_copy) {
            copies.push_back(c);
        } else if (!skip) {
            chosen.push_back(c);
            if (chosen.size() == m) {
                break;
            }
        }
    }
    for (std::size_t i = 0; i < copies.size() && chosen.size() < m; ++i) {
        chosen.push_back(copies[i]);
    }
    candidates = std::move(chosen);
}

void HnswIndex::connect(NodeId node, NodeId new_node, float distance, int level) {
    std::lock_guard lock(node_locks_[node]);
    NodeId* block = links(node, level);
    const std::size_t count = block[0];
    for (std::size_t i = 0; i < count; ++i) {
        if (block[1 + i] == new_node) {
            return;
        }
    }
    if (count < max_links(level)) {
        block[1 + count] = new_node;
        block[0] = static_cast<NodeId>(count + 1);
        return;
    }

    // The list is full: choose the best set out of the old links plus the new one.
    std::vector<Candidate> candidates;
    candidates.reserve(count + 1);
    candidates.push_back({distance, new_node});
    for (std::size_t i = 0; i < count; ++i) {
        const NodeId other = block[1 + i];
        candidates.push_back({distance_(vector_of(node), vector_of(other), dim_), other});
    }
    std::sort(candidates.begin(), candidates.end(), closer<Candidate>);
    select_neighbors(candidates, max_links(level));
    block[0] = static_cast<NodeId>(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        block[1 + i] = candidates[i].id;
    }
}

void HnswIndex::search(std::span<const float> query, std::size_t k, std::size_t ef_search,
                       std::span<RowId> ids, std::span<float> distances) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("query size does not match dim");
    }
    if (ids.size() < k || distances.size() < k) {
        throw std::invalid_argument("output buffers are smaller than k");
    }
    if (k == 0) {
        return;
    }

    const float* q = query.data();
    std::vector<float> normalized;
    if (metric_ == Metric::Cosine) {
        normalized.assign(query.begin(), query.end());
        normalize(normalized.data(), dim_);
        q = normalized.data();
    }

    std::vector<Candidate> found;
    const std::uint64_t entry = entry_.load(std::memory_order_acquire);
    if (entry != kNoEntry) {
        std::vector<NodeId> scratch;
        Candidate current{distance_(q, vector_of(entry_node(entry)), dim_), entry_node(entry)};
        for (int l = entry_level(entry); l > 0; --l) {
            greedy_search(q, current, l, scratch);
        }
        VisitedHandle visited(*visited_pool_);
        found = search_layer(q, {current}, std::max(ef_search, k), 0, *visited);
    }

    for (std::size_t i = 0; i < k; ++i) {
        if (i < found.size()) {
            ids[i] = found[i].id;
            distances[i] = found[i].distance;
        } else {
            ids[i] = kNoRow;
            distances[i] = std::numeric_limits<float>::infinity();
        }
    }
}

void HnswIndex::search_batch(std::span<const float> queries, std::size_t k, std::size_t ef_search,
                             std::span<RowId> ids, std::span<float> distances,
                             std::size_t threads) const {
    if (queries.size() % dim_ != 0) {
        throw std::invalid_argument("query data size is not a multiple of dim");
    }
    const std::size_t nq = queries.size() / dim_;
    if (ids.size() < nq * k || distances.size() < nq * k) {
        throw std::invalid_argument("output buffers are smaller than queries * k");
    }
    parallel_for(nq, threads, [&](std::size_t q) {
        search(queries.subspan(q * dim_, dim_), k, ef_search, ids.subspan(q * k, k),
               distances.subspan(q * k, k));
    });
}

HnswStats HnswIndex::stats() const {
    HnswStats s;
    s.size = size();
    const std::uint64_t entry = entry_.load(std::memory_order_acquire);
    s.max_level = entry == kNoEntry ? -1 : entry_level(entry);
    const std::size_t level_count = s.max_level < 0 ? 0 : static_cast<std::size_t>(s.max_level) + 1;
    s.nodes_per_level.assign(level_count, 0);

    std::size_t total_links0 = 0;
    std::size_t upper_slots = 0;
    for (NodeId node = 0; node < s.size; ++node) {
        for (int l = 0; l <= levels_[node]; ++l) {
            ++s.nodes_per_level[static_cast<std::size_t>(l)];
        }
        const std::size_t links0 = links(node, 0)[0];
        total_links0 += links0;
        s.max_links_level0 = std::max(s.max_links_level0, links0);
        upper_slots += upper_links_[node].size();
    }
    if (s.size > 0) {
        s.avg_links_level0 = static_cast<double>(total_links0) / static_cast<double>(s.size);
    }
    s.memory_bytes = vectors_.capacity() * sizeof(float) +
                     (level0_links_.size() + upper_slots) * sizeof(NodeId) + levels_.size() +
                     capacity_ * (sizeof(std::mutex) + sizeof(std::vector<NodeId>));
    return s;
}

void HnswIndex::save(const std::string& path) const {
    FilePtr f(std::fopen(path.c_str(), "wb"));
    if (!f) {
        throw std::runtime_error("cannot open '" + path + "' for writing");
    }
    const std::size_t n = size();
    write_bytes(f.get(), kMagic, sizeof(kMagic));
    write_value(f.get(), kFileVersion);
    write_value(f.get(), static_cast<std::uint32_t>(metric_));
    write_value(f.get(), static_cast<std::uint64_t>(dim_));
    write_value(f.get(), static_cast<std::uint64_t>(params_.M));
    write_value(f.get(), static_cast<std::uint64_t>(params_.ef_construction));
    write_value(f.get(), params_.seed);
    write_value(f.get(), static_cast<std::uint64_t>(n));
    write_value(f.get(), entry_.load(std::memory_order_acquire));

    for (NodeId node = 0; node < n; ++node) {
        write_bytes(f.get(), vector_of(node), dim_ * sizeof(float));
    }
    write_bytes(f.get(), levels_.data(), n);
    write_bytes(f.get(), level0_links_.data(), n * (1 + max_links_level0_) * sizeof(NodeId));
    for (NodeId node = 0; node < n; ++node) {
        write_bytes(f.get(), upper_links_[node].data(), upper_links_[node].size() * sizeof(NodeId));
    }
    if (std::fclose(f.release()) != 0) {
        throw std::runtime_error("close failed for '" + path + "'");
    }
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path, std::size_t capacity) {
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    char magic[sizeof(kMagic)];
    read_bytes(f.get(), magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        read_value<std::uint32_t>(f.get()) != kFileVersion) {
        throw std::runtime_error("'" + path + "' is not a StrataDB HNSW file");
    }
    const auto metric_value = read_value<std::uint32_t>(f.get());
    const auto dim = read_value<std::uint64_t>(f.get());
    HnswParams params;
    params.M = read_value<std::uint64_t>(f.get());
    params.ef_construction = read_value<std::uint64_t>(f.get());
    params.seed = read_value<std::uint64_t>(f.get());
    const auto n = read_value<std::uint64_t>(f.get());
    const auto entry = read_value<std::uint64_t>(f.get());
    if (metric_value > static_cast<std::uint32_t>(Metric::Cosine) || dim == 0 || dim > 65536 ||
        params.M < 2 || params.M > 1024 || n >= std::numeric_limits<NodeId>::max()) {
        throw std::runtime_error("'" + path + "' has an invalid header");
    }

    auto index = std::make_unique<HnswIndex>(dim, static_cast<Metric>(metric_value),
                                             std::max<std::size_t>({capacity, n, 1}), params);
    for (NodeId node = 0; node < n; ++node) {
        read_bytes(f.get(), index->vector_of(node), dim * sizeof(float));
    }
    read_bytes(f.get(), index->levels_.data(), n);
    read_bytes(f.get(), index->level0_links_.data(),
               n * (1 + index->max_links_level0_) * sizeof(NodeId));
    for (NodeId node = 0; node < n; ++node) {
        const int level = index->levels_[node];
        if (level > kMaxLevel) {
            throw std::runtime_error("'" + path + "' has an invalid node level");
        }
        if (level > 0) {
            index->upper_links_[node].resize(static_cast<std::size_t>(level) * (1 + params.M));
            read_bytes(f.get(), index->upper_links_[node].data(),
                       index->upper_links_[node].size() * sizeof(NodeId));
        }
    }

    // Check every link so a damaged file can't make a search read out of bounds.
    for (NodeId node = 0; node < n; ++node) {
        for (int l = 0; l <= index->levels_[node]; ++l) {
            const NodeId* block = index->links(node, l);
            bool ok = block[0] <= index->max_links(l);
            for (NodeId i = 0; ok && i < block[0]; ++i) {
                const NodeId other = block[1 + i];
                ok = other < n && index->levels_[other] >= l;
            }
            if (!ok) {
                throw std::runtime_error("'" + path + "' has invalid links");
            }
        }
    }
    if (n > 0 && (entry == kNoEntry || entry_node(entry) >= n ||
                  entry_level(entry) != index->levels_[entry_node(entry)])) {
        throw std::runtime_error("'" + path + "' has an invalid entry point");
    }
    index->count_.store(n);
    index->entry_.store(n > 0 ? entry : kNoEntry);
    return index;
}

}  // namespace strata
