#include "strata/flat_index.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "strata/distance.h"

namespace strata {
namespace {

// Rows are padded to a multiple of 16 floats (64 bytes). The padding is zero,
// so it never changes a distance.
constexpr std::size_t kFloatsPerCacheLine = 64 / sizeof(float);

std::size_t round_up_to_cache_line(std::size_t dim) {
    return (dim + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine;
}

// Keeps the k best (smallest distance) candidates seen so far.
// The worst of them sits on top of a max-heap, so a new candidate only
// has to beat that one.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

    void push(float distance, RowId id) {
        const Item item{distance, id};
        if (heap_.size() < k_) {
            heap_.push_back(item);
            std::push_heap(heap_.begin(), heap_.end(), closer);
        } else if (closer(item, heap_.front())) {
            std::pop_heap(heap_.begin(), heap_.end(), closer);
            heap_.back() = item;
            std::push_heap(heap_.begin(), heap_.end(), closer);
        }
    }

    // Writes results closest first and fills unused slots.
    void write(RowId* ids, float* distances) {
        std::sort_heap(heap_.begin(), heap_.end(), closer);
        for (std::size_t i = 0; i < k_; ++i) {
            if (i < heap_.size()) {
                ids[i] = heap_[i].id;
                distances[i] = heap_[i].distance;
            } else {
                ids[i] = kNoRow;
                distances[i] = std::numeric_limits<float>::infinity();
            }
        }
    }

private:
    struct Item {
        float distance;
        RowId id;
    };

    // True if a is closer than b. Equal distances are ordered by row id,
    // so results never depend on the order rows were scanned in.
    static bool closer(const Item& a, const Item& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    }

    std::size_t k_;
    std::vector<Item> heap_;
};

}  // namespace

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), stride_(round_up_to_cache_line(dim)), metric_(metric) {
    if (dim == 0) {
        throw std::invalid_argument("dim must be > 0");
    }
}

void FlatIndex::add(std::span<const float> vectors) {
    if (vectors.size() % dim_ != 0) {
        throw std::invalid_argument("vector data size is not a multiple of dim");
    }
    const std::size_t n = vectors.size() / dim_;
    vectors_.resize((count_ + n) * stride_);

    for (std::size_t i = 0; i < n; ++i) {
        float* dst = vectors_.data() + (count_ + i) * stride_;
        std::copy_n(vectors.data() + i * dim_, dim_, dst);
        if (metric_ == Metric::Cosine) {
            normalize(dst, dim_);
        }
    }
    count_ += n;
}

void FlatIndex::search(std::span<const float> query, std::size_t k, std::span<RowId> ids,
                       std::span<float> distances) const {
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

    const DistanceFn distance = distance_function(metric_);
    TopK best(k);
    for (std::size_t i = 0; i < count_; ++i) {
        best.push(distance(q, row(i), dim_), static_cast<RowId>(i));
    }
    best.write(ids.data(), distances.data());
}

void FlatIndex::search_batch(std::span<const float> queries, std::size_t k, std::span<RowId> ids,
                             std::span<float> distances, std::size_t threads) const {
    if (queries.size() % dim_ != 0) {
        throw std::invalid_argument("query data size is not a multiple of dim");
    }
    const std::size_t nq = queries.size() / dim_;
    if (ids.size() < nq * k || distances.size() < nq * k) {
        throw std::invalid_argument("output buffers are smaller than queries * k");
    }
    if (nq == 0 || k == 0) {
        return;
    }

    if (threads == 0) {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }
    threads = std::min(threads, nq);

    // Threads take small batches of queries from a shared counter. Fast and slow
    // cores then finish at about the same time instead of waiting on each other.
    constexpr std::size_t kBatch = 8;
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        while (true) {
            const std::size_t start = next.fetch_add(kBatch);
            if (start >= nq) {
                return;
            }
            const std::size_t end = std::min(start + kBatch, nq);
            for (std::size_t q = start; q < end; ++q) {
                search(queries.subspan(q * dim_, dim_), k, ids.subspan(q * k, k),
                       distances.subspan(q * k, k));
            }
        }
    };

    std::vector<std::jthread> pool;
    pool.reserve(threads - 1);
    for (std::size_t t = 1; t < threads; ++t) {
        pool.emplace_back(worker);
    }
    worker();  // the calling thread helps too
}

}  // namespace strata
