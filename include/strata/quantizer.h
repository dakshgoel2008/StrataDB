#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strata {

// SQ8: stores every value in one byte instead of four.
// Each dimension gets its own range [min, max], learned from sample vectors and
// split into 256 even steps. value = min + code * scale.
class ScalarQuantizer {
public:
    ScalarQuantizer() = default;

    // sample holds rows of dim floats.
    static ScalarQuantizer train(std::span<const float> sample, std::size_t dim);
    // Rebuilds a trained quantizer from its saved ranges.
    static ScalarQuantizer from_ranges(std::vector<float> mins, std::vector<float> scales);

    std::size_t dim() const noexcept { return dim_; }
    std::size_t code_size() const noexcept { return dim_; }

    // Values outside the learned range are clamped to it.
    void encode(const float* vector, std::uint8_t* code) const noexcept;
    void decode(const std::uint8_t* code, float* vector) const noexcept;

    // A query rewritten so its distance to any code is one quick pass over the code.
    //   squared L2  = constant + sum of weight[i] * (value[i] - code[i])^2
    //   dot product = constant + sum of value[i] * code[i]
    struct PreparedQuery {
        std::vector<float> value;
        std::vector<float> weight;
        float constant = 0;
    };
    void prepare_l2(const float* query, PreparedQuery& out) const;
    void prepare_dot(const float* query, PreparedQuery& out) const;

    const std::vector<float>& mins() const noexcept { return min_; }
    const std::vector<float>& scales() const noexcept { return scale_; }

private:
    std::size_t dim_ = 0;
    std::vector<float> min_;
    std::vector<float> scale_;
};

// PQ: cuts a vector into m equal pieces and replaces each piece with the id (one
// byte) of the closest of 256 centroids learned for that piece with k-means.
// A 768-dim vector with m = 96 takes 96 bytes instead of 3072.
class ProductQuantizer {
public:
    static constexpr std::size_t kCentroids = 256;

    struct TrainOptions {
        std::size_t m = 8;            // number of pieces; must divide dim
        std::size_t iterations = 20;  // k-means rounds per piece
        std::uint64_t seed = 1;
        std::size_t threads = 0;  // 0 = all cores
    };

    ProductQuantizer() = default;

    // sample holds rows of dim floats; needs at least 256 rows.
    static ProductQuantizer train(std::span<const float> sample, std::size_t dim,
                                  const TrainOptions& options);
    // Rebuilds a trained quantizer from its saved centroids (256 * dim floats).
    static ProductQuantizer from_centroids(std::size_t dim, std::size_t m,
                                           std::vector<float> centroids);

    std::size_t dim() const noexcept { return dim_; }
    std::size_t m() const noexcept { return m_; }
    std::size_t sub_dim() const noexcept { return sub_dim_; }
    std::size_t code_size() const noexcept { return m_; }

    void encode(const float* vector, std::uint8_t* code) const noexcept;
    void decode(const std::uint8_t* code, float* vector) const noexcept;

    // Fills table (m * 256 floats) with, for each piece and centroid, the squared
    // distance (use_dot = false) or dot product (use_dot = true) between the query
    // piece and that centroid. Adding up the entries a code picks gives the squared
    // distance or dot product between the query and the encoded vector.
    void compute_table(const float* query, bool use_dot, float* table) const noexcept;

    // The 256 centroids of one piece, back to back.
    const float* centroids(std::size_t piece) const noexcept {
        return centroids_.data() + piece * kCentroids * sub_dim_;
    }
    const std::vector<float>& all_centroids() const noexcept { return centroids_; }

private:
    std::size_t dim_ = 0;
    std::size_t m_ = 0;
    std::size_t sub_dim_ = 0;
    std::vector<float> centroids_;  // m * 256 * sub_dim
};

// Sum of the table entries picked by a PQ code (see compute_table).
float pq_table_sum(const float* table, const std::uint8_t* code, std::size_t m) noexcept;

// k-means with k-means++ starting points. points holds n rows of d floats.
// Returns k * d centroid values. Same seed gives the same result.
std::vector<float> kmeans(const float* points, std::size_t n, std::size_t d, std::size_t k,
                          std::size_t iterations, std::uint64_t seed);

}  // namespace strata
