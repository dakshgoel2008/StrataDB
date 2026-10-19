// Writes small valid files to start the fuzzers from: a log file with a few
// records and a few snapshots of different kinds.
// Usage: make_seeds <output dir>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "strata/database.h"
#include "strata/file_system.h"
#include "strata/hnsw_index.h"
#include "strata/quantizer.h"

namespace {

std::vector<float> random_vectors(std::size_t n, std::size_t dim) {
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (float& x : v) {
        x = dist(rng);
    }
    return v;
}

void save(const strata::HnswIndex& index, const std::string& path) {
    index.save_snapshot(strata::local_file_system(), path, index.save_point(), 3);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: make_seeds <output dir>\n");
        return 2;
    }
    const std::string out = argv[1];
    std::filesystem::create_directories(out + "/wal");
    std::filesystem::create_directories(out + "/snapshot");

    std::vector<std::uint8_t> log;
    const auto vectors = random_vectors(6, 4);
    for (std::uint64_t i = 0; i < 3; ++i) {
        const auto payload = strata::encode_insert(
            i * 2, std::span(vectors).subspan(static_cast<std::size_t>(i) * 8, 8), 4);
        strata::encode_wal_record(i + 1, strata::WalRecordType::Insert, payload, log);
    }
    auto file = strata::local_file_system().create(out + "/wal/three_records");
    file->append(log.data(), log.size());

    const auto base = random_vectors(300, 8);
    strata::HnswIndex plain(8, strata::Metric::L2, 300, {4, 16, 1});
    plain.add(std::span(base).first(40 * 8));
    save(plain, out + "/snapshot/float");

    strata::HnswIndex sq8(8, strata::Metric::Cosine, 300, {4, 16, 1},
                          strata::ScalarQuantizer::train(base, 8));
    sq8.add(std::span(base).first(40 * 8));
    save(sq8, out + "/snapshot/sq8");

    strata::HnswIndex pq(8, strata::Metric::InnerProduct, 300, {4, 16, 1},
                         strata::ProductQuantizer::train(base, 8, {2, 5, 1, 1}));
    pq.add(std::span(base).first(40 * 8));
    save(pq, out + "/snapshot/pq");
    return 0;
}
