// Finds the exact k nearest neighbors of every query by checking all vectors.
// Writes <out>.ivecs (neighbor ids) and <out>_distances.fvecs.
//
// Example:
//   compute_groundtruth --base data/sift/base.fvecs --query data/sift/query.fvecs
//                       --metric l2 --k 100 --out data/sift/groundtruth_strata

#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bench/common/timer.h"
#include "bench/common/vecs_io.h"
#include "strata/flat_index.h"
#include "strata/metric.h"

namespace {

namespace sb = strata::bench;

constexpr std::size_t kLoadChunk = 100'000;  // base rows read at a time

struct Options {
    std::string base;
    std::string query;
    std::string out;
    strata::Metric metric = strata::Metric::L2;
    std::size_t k = 100;
    std::size_t threads = 0;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --base FILE.fvecs --query FILE.fvecs --metric l2|ip|cosine\n"
                 "          --out PREFIX [--k 100] [--threads N]\n",
                 prog);
}

Options parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        }
        args[argv[i]] = argv[i + 1];
    }

    Options opt;
    for (const auto& [key, value] : args) {
        if (key == "--base") {
            opt.base = value;
        } else if (key == "--query") {
            opt.query = value;
        } else if (key == "--out") {
            opt.out = value;
        } else if (key == "--metric") {
            const auto m = strata::parse_metric(value);
            if (!m) {
                throw std::invalid_argument("unknown metric '" + value + "'");
            }
            opt.metric = *m;
        } else if (key == "--k") {
            opt.k = std::stoul(value);
        } else if (key == "--threads") {
            opt.threads = std::stoul(value);
        } else {
            throw std::invalid_argument("unknown option " + key);
        }
    }
    if (opt.base.empty() || opt.query.empty() || opt.out.empty() || opt.k == 0) {
        throw std::invalid_argument("--base, --query and --out are required, and k must be > 0");
    }
    return opt;
}

void run(const Options& opt) {
    const sb::VecsShape shape = sb::inspect_vecs(opt.base, sizeof(float));
    std::printf("base   : %zu x %zu, metric %s\n", shape.rows, shape.dim,
                std::string(strata::metric_name(opt.metric)).c_str());
    if (shape.rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("too many rows for an .ivecs file");
    }

    // Load in chunks so only one copy of the data is ever in memory.
    sb::Stopwatch load_timer;
    strata::FlatIndex index(shape.dim, opt.metric);
    index.reserve(shape.rows);
    for (std::size_t start = 0; start < shape.rows; start += kLoadChunk) {
        const auto chunk = sb::read_fvecs(opt.base, kLoadChunk, start);
        index.add(chunk.data);
        std::printf("\rloading: %zu / %zu", index.size(), shape.rows);
        std::fflush(stdout);
    }
    std::printf("  (%.1f s)\n", load_timer.elapsed_s());

    const auto queries = sb::read_fvecs(opt.query);
    if (queries.dim != shape.dim) {
        throw std::runtime_error("query dim does not match base dim");
    }
    std::printf("queries: %zu, k = %zu\n", queries.rows, opt.k);

    std::vector<strata::RowId> ids(queries.rows * opt.k);
    std::vector<float> distances(queries.rows * opt.k);
    sb::Stopwatch search_timer;
    index.search_batch(queries.data, opt.k, ids, distances, opt.threads);
    const double seconds = search_timer.elapsed_s();
    std::printf("search : %.1f s (%.0f queries/s)\n", seconds,
                static_cast<double>(queries.rows) / seconds);

    sb::Matrix<std::int32_t> id_matrix{queries.rows, opt.k, std::vector<std::int32_t>(ids.size())};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        id_matrix.data[i] = static_cast<std::int32_t>(ids[i]);
    }
    sb::write_ivecs(opt.out + ".ivecs", id_matrix);
    sb::write_fvecs(opt.out + "_distances.fvecs", {queries.rows, opt.k, std::move(distances)});
    std::printf("wrote  : %s.ivecs and %s_distances.fvecs\n", opt.out.c_str(), opt.out.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse_args(argc, argv));
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
