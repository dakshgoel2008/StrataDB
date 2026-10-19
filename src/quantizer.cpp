#include "strata/quantizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "strata/parallel.h"

namespace strata {
namespace {

float squared_distance(const float* a, const float* b, std::size_t d) {
    float sum = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float dot(const float* a, const float* b, std::size_t d) {
    float sum = 0;
    for (std::size_t i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Index of the centroid closest to point.
std::size_t closest(const float* point, const float* centroids, std::size_t k, std::size_t d,
                    float* best_distance = nullptr) {
    std::size_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < k; ++c) {
        const float dist = squared_distance(point, centroids + c * d, d);
        if (dist < best_d) {
            best_d = dist;
            best = c;
        }
    }
    if (best_distance != nullptr) {
        *best_distance = best_d;
    }
    return best;
}

}  // namespace

ScalarQuantizer ScalarQuantizer::train(std::span<const float> sample, std::size_t dim) {
    if (dim == 0 || sample.empty() || sample.size() % dim != 0) {
        throw std::invalid_argument("SQ8 training needs at least one vector of size dim");
    }
    const std::size_t n = sample.size() / dim;
    ScalarQuantizer sq;
    sq.dim_ = dim;
    sq.min_.assign(dim, std::numeric_limits<float>::infinity());
    std::vector<float> max(dim, -std::numeric_limits<float>::infinity());
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < dim; ++i) {
            const float v = sample[r * dim + i];
            sq.min_[i] = std::min(sq.min_[i], v);
            max[i] = std::max(max[i], v);
        }
    }
    sq.scale_.resize(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        sq.scale_[i] = (max[i] - sq.min_[i]) / 255.0f;
    }
    return sq;
}

ScalarQuantizer ScalarQuantizer::from_ranges(std::vector<float> mins, std::vector<float> scales) {
    if (mins.empty() || mins.size() != scales.size()) {
        throw std::invalid_argument("SQ8 ranges need one min and one scale per dimension");
    }
    ScalarQuantizer sq;
    sq.dim_ = mins.size();
    sq.min_ = std::move(mins);
    sq.scale_ = std::move(scales);
    return sq;
}

void ScalarQuantizer::encode(const float* vector, std::uint8_t* code) const noexcept {
    for (std::size_t i = 0; i < dim_; ++i) {
        if (scale_[i] == 0) {
            code[i] = 0;
            continue;
        }
        const float step = std::round((vector[i] - min_[i]) / scale_[i]);
        code[i] = static_cast<std::uint8_t>(std::clamp(step, 0.0f, 255.0f));
    }
}

void ScalarQuantizer::decode(const std::uint8_t* code, float* vector) const noexcept {
    for (std::size_t i = 0; i < dim_; ++i) {
        vector[i] = min_[i] + static_cast<float>(code[i]) * scale_[i];
    }
}

// (q - min - c * s)^2 = s^2 * ((q - min) / s - c)^2, so each code value only
// needs one subtract and one multiply-add.
void ScalarQuantizer::prepare_l2(const float* query, PreparedQuery& out) const {
    out.value.assign(dim_, 0);
    out.weight.assign(dim_, 0);
    out.constant = 0;
    for (std::size_t i = 0; i < dim_; ++i) {
        const float shifted = query[i] - min_[i];
        if (scale_[i] == 0) {
            out.constant += shifted * shifted;  // every code decodes to min here
        } else {
            out.value[i] = shifted / scale_[i];
            out.weight[i] = scale_[i] * scale_[i];
        }
    }
}

// q . (min + c * s) = q . min + sum of (q * s) * c
void ScalarQuantizer::prepare_dot(const float* query, PreparedQuery& out) const {
    out.value.resize(dim_);
    out.weight.clear();
    out.constant = 0;
    for (std::size_t i = 0; i < dim_; ++i) {
        out.value[i] = query[i] * scale_[i];
        out.constant += query[i] * min_[i];
    }
}

std::vector<float> kmeans(const float* points, std::size_t n, std::size_t d, std::size_t k,
                          std::size_t iterations, std::uint64_t seed) {
    if (n < k || k == 0 || d == 0) {
        throw std::invalid_argument("k-means needs at least k points");
    }
    std::mt19937_64 rng(seed);
    std::vector<float> centroids(k * d);

    // k-means++: each new starting centroid is picked with probability proportional
    // to its squared distance from the closest centroid chosen so far.
    std::vector<float> nearest(n, std::numeric_limits<float>::infinity());
    std::size_t pick = std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
    for (std::size_t c = 0; c < k; ++c) {
        std::copy_n(points + pick * d, d, centroids.data() + c * d);
        double total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            nearest[i] =
                std::min(nearest[i], squared_distance(points + i * d, centroids.data() + c * d, d));
            total += nearest[i];
        }
        if (c + 1 == k) {
            break;
        }
        if (total == 0) {
            pick = std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
            continue;
        }
        double target = std::uniform_real_distribution<double>(0, total)(rng);
        pick = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
            target -= nearest[i];
            if (target <= 0) {
                pick = i;
                break;
            }
        }
    }

    // Lloyd rounds: assign every point to its closest centroid, then move each
    // centroid to the mean of its points.
    std::vector<std::size_t> assignment(n, k);
    std::vector<float> point_distance(n);
    std::vector<double> sums(k * d);
    std::vector<std::size_t> counts(k);
    for (std::size_t round = 0; round < iterations; ++round) {
        bool changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c =
                closest(points + i * d, centroids.data(), k, d, &point_distance[i]);
            changed |= c != assignment[i];
            assignment[i] = c;
        }
        if (!changed) {
            break;
        }

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c = assignment[i];
            ++counts[c];
            for (std::size_t j = 0; j < d; ++j) {
                sums[c * d + j] += points[i * d + j];
            }
        }
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                // An empty cluster takes over the point that is worst served right now.
                const auto far = static_cast<std::size_t>(
                    std::max_element(point_distance.begin(), point_distance.end()) -
                    point_distance.begin());
                std::copy_n(points + far * d, d, centroids.data() + c * d);
                point_distance[far] = 0;
                continue;
            }
            for (std::size_t j = 0; j < d; ++j) {
                centroids[c * d + j] =
                    static_cast<float>(sums[c * d + j] / static_cast<double>(counts[c]));
            }
        }
    }
    return centroids;
}

ProductQuantizer ProductQuantizer::train(std::span<const float> sample, std::size_t dim,
                                         const TrainOptions& options) {
    if (dim == 0 || options.m == 0 || dim % options.m != 0) {
        throw std::invalid_argument("PQ needs m > 0 pieces that divide dim evenly");
    }
    if (sample.size() % dim != 0 || sample.size() / dim < kCentroids) {
        throw std::invalid_argument("PQ training needs at least 256 vectors");
    }
    const std::size_t n = sample.size() / dim;

    ProductQuantizer pq;
    pq.dim_ = dim;
    pq.m_ = options.m;
    pq.sub_dim_ = dim / options.m;
    pq.centroids_.resize(pq.m_ * kCentroids * pq.sub_dim_);

    // Every piece is trained on its own, so pieces run in parallel. Each piece has
    // its own seed, so the result doesn't depend on the thread count.
    parallel_for(
        pq.m_, options.threads,
        [&](std::size_t piece) {
            std::vector<float> slice(n * pq.sub_dim_);
            for (std::size_t r = 0; r < n; ++r) {
                std::copy_n(sample.data() + r * dim + piece * pq.sub_dim_, pq.sub_dim_,
                            slice.data() + r * pq.sub_dim_);
            }
            const std::vector<float> c = kmeans(slice.data(), n, pq.sub_dim_, kCentroids,
                                                options.iterations, options.seed + piece);
            std::copy(c.begin(), c.end(),
                      pq.centroids_.begin() + static_cast<long>(piece * kCentroids * pq.sub_dim_));
        },
        1);
    return pq;
}

ProductQuantizer ProductQuantizer::from_centroids(std::size_t dim, std::size_t m,
                                                  std::vector<float> centroids) {
    if (dim == 0 || m == 0 || dim % m != 0 || centroids.size() != kCentroids * dim) {
        throw std::invalid_argument("PQ centroids don't match dim and m");
    }
    ProductQuantizer pq;
    pq.dim_ = dim;
    pq.m_ = m;
    pq.sub_dim_ = dim / m;
    pq.centroids_ = std::move(centroids);
    return pq;
}

void ProductQuantizer::encode(const float* vector, std::uint8_t* code) const noexcept {
    for (std::size_t piece = 0; piece < m_; ++piece) {
        code[piece] = static_cast<std::uint8_t>(
            closest(vector + piece * sub_dim_, centroids(piece), kCentroids, sub_dim_));
    }
}

void ProductQuantizer::decode(const std::uint8_t* code, float* vector) const noexcept {
    for (std::size_t piece = 0; piece < m_; ++piece) {
        std::copy_n(centroids(piece) + code[piece] * sub_dim_, sub_dim_, vector + piece * sub_dim_);
    }
}

void ProductQuantizer::compute_table(const float* query, bool use_dot,
                                     float* table) const noexcept {
    for (std::size_t piece = 0; piece < m_; ++piece) {
        const float* q = query + piece * sub_dim_;
        const float* c = centroids(piece);
        float* row = table + piece * kCentroids;
        for (std::size_t j = 0; j < kCentroids; ++j) {
            row[j] = use_dot ? dot(q, c + j * sub_dim_, sub_dim_)
                             : squared_distance(q, c + j * sub_dim_, sub_dim_);
        }
    }
}

float pq_table_sum(const float* table, const std::uint8_t* code, std::size_t m) noexcept {
    constexpr std::size_t K = ProductQuantizer::kCentroids;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= m; i += 4) {
        s0 += table[i * K + code[i]];
        s1 += table[(i + 1) * K + code[i + 1]];
        s2 += table[(i + 2) * K + code[i + 2]];
        s3 += table[(i + 3) * K + code[i + 3]];
    }
    for (; i < m; ++i) {
        s0 += table[i * K + code[i]];
    }
    return (s0 + s1) + (s2 + s3);
}

}  // namespace strata
