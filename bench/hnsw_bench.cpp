// Builds an HNSW index on a prepared dataset and measures recall vs. speed,
// for StrataDB and for hnswlib (compiled into this program, same settings).
// Results go to bench/results/baselines/<dataset>-<library>.json.
//
// Example:
//   ./build/release/bench/hnsw_bench --dataset sift --lib both

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bench/common/metrics.h"
#include "bench/common/timer.h"
#include "bench/common/vecs_io.h"
#include "hnswlib/hnswlib.h"
#include "strata/distance.h"
#include "strata/hnsw_index.h"
#include "strata/parallel.h"
#include "strata/version.h"

namespace {

namespace sb = strata::bench;

constexpr std::size_t kLoadChunk = 100'000;
constexpr std::size_t kWarmupQueries = 200;
constexpr std::array<std::size_t, 12> kEfSweep = {10, 16,  24,  32,  48,  64,
                                                  96, 128, 192, 256, 384, 512};

struct Options {
    std::string dataset;
    std::string lib = "both";
    std::string data_dir = "data";
    std::string results_dir = "bench/results/baselines";
    std::size_t M = 16;
    std::size_t ef_construction = 200;
    std::size_t k = 10;
    std::size_t threads = 0;
    int search_cpu = 2;
    std::size_t max_queries = 0;
    std::string simd;  // empty = fastest this CPU supports
    std::size_t repeats = 3;
};

// What a library needs to provide to be measured.
class Searcher {
public:
    virtual ~Searcher() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual void add(const float* data, std::size_t n, std::size_t first_id,
                     std::size_t threads) = 0;
    virtual void set_ef(std::size_t ef) = 0;
    virtual void search(const float* query, std::size_t k, std::int32_t* ids) = 0;
    virtual std::string extra_info() const { return ""; }
};

class StrataSearcher : public Searcher {
public:
    StrataSearcher(std::size_t dim, strata::Metric metric, std::size_t capacity, const Options& opt)
        : index_(dim, metric, capacity, strata::HnswParams{opt.M, opt.ef_construction, 100}),
          ids_(opt.k),
          distances_(opt.k) {}

    std::string name() const override {
        return "strata-" + std::string(strata::simd_level_name(strata::active_simd_level()));
    }
    std::string version() const override { return std::string(strata::version()); }

    void add(const float* data, std::size_t n, std::size_t, std::size_t threads) override {
        index_.add({data, n * index_.dim()}, threads);
    }

    void set_ef(std::size_t ef) override { ef_ = ef; }

    void search(const float* query, std::size_t k, std::int32_t* ids) override {
        index_.search({query, index_.dim()}, k, ef_, ids_, distances_);
        for (std::size_t i = 0; i < k; ++i) {
            ids[i] = static_cast<std::int32_t>(ids_[i]);
        }
    }

    std::string extra_info() const override {
        const auto s = index_.stats();
        std::ostringstream out;
        out << "levels " << s.max_level + 1 << ", avg links on layer 0 " << s.avg_links_level0
            << ", memory " << s.memory_bytes / (1 << 20) << " MiB";
        return out.str();
    }

private:
    strata::HnswIndex index_;
    std::size_t ef_ = 10;
    std::vector<strata::RowId> ids_;
    std::vector<float> distances_;
};

class HnswlibSearcher : public Searcher {
public:
    HnswlibSearcher(std::size_t dim, strata::Metric metric, std::size_t capacity,
                    const Options& opt)
        : dim_(dim), normalize_(metric == strata::Metric::Cosine), query_(dim) {
        if (metric == strata::Metric::L2) {
            space_ = std::make_unique<hnswlib::L2Space>(dim);
        } else {
            space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        }
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), capacity, opt.M,
                                                                   opt.ef_construction, 100);
    }

    std::string name() const override { return "hnswlib-cpp"; }
    std::string version() const override { return "0.8.0"; }

    void add(const float* data, std::size_t n, std::size_t first_id, std::size_t threads) override {
        strata::parallel_for(n, threads, [&](std::size_t i) {
            const float* v = data + i * dim_;
            std::vector<float> copy;
            if (normalize_) {
                copy.assign(v, v + dim_);
                strata::normalize(copy.data(), dim_);
                v = copy.data();
            }
            index_->addPoint(v, first_id + i);
        });
    }

    void set_ef(std::size_t ef) override { index_->setEf(ef); }

    void search(const float* query, std::size_t k, std::int32_t* ids) override {
        const float* q = query;
        if (normalize_) {
            std::copy_n(query, dim_, query_.begin());
            strata::normalize(query_.data(), dim_);
            q = query_.data();
        }
        auto result = index_->searchKnn(q, k);
        std::fill(ids, ids + k, -1);
        // The result comes back farthest first.
        for (std::size_t i = result.size(); i > 0; --i) {
            ids[i - 1] = static_cast<std::int32_t>(result.top().second);
            result.pop();
        }
    }

private:
    std::size_t dim_;
    bool normalize_;
    std::vector<float> query_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// Pulls "metric" out of meta.json without a full JSON parser.
strata::Metric read_metric(const std::string& meta_path) {
    const std::string text = read_file(meta_path);
    const std::size_t key = text.find("\"metric\"");
    const std::size_t open = text.find('"', text.find(':', key) + 1);
    const std::size_t close = text.find('"', open + 1);
    if (key == std::string::npos || open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("no metric in '" + meta_path + "'");
    }
    const auto metric = strata::parse_metric(text.substr(open + 1, close - open - 1));
    if (!metric) {
        throw std::runtime_error("unknown metric in '" + meta_path + "'");
    }
    return *metric;
}

std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            return line.substr(line.find(':') + 2);
        }
    }
    return "unknown";
}

void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<std::size_t>(cpu), &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        throw std::runtime_error("cannot pin to CPU " + std::to_string(cpu));
    }
}

struct Point {
    std::size_t ef;
    double recall;
    double qps;
    sb::LatencySummary latency;
};

void run(const std::string& lib, const Options& opt) {
    const std::string folder = opt.data_dir + "/" + opt.dataset;
    const strata::Metric metric = read_metric(folder + "/meta.json");
    const std::string base_path = folder + "/base.fvecs";
    const sb::VecsShape shape = sb::inspect_vecs(base_path, sizeof(float));

    std::unique_ptr<Searcher> searcher;
    if (lib == "strata") {
        searcher = std::make_unique<StrataSearcher>(shape.dim, metric, shape.rows, opt);
    } else {
        searcher = std::make_unique<HnswlibSearcher>(shape.dim, metric, shape.rows, opt);
    }
    std::printf("[%s / %s %s] building on %zu x %zu (M=%zu, ef_construction=%zu)\n",
                opt.dataset.c_str(), searcher->name().c_str(), searcher->version().c_str(),
                shape.rows, shape.dim, opt.M, opt.ef_construction);

    sb::Stopwatch build_timer;
    for (std::size_t start = 0; start < shape.rows; start += kLoadChunk) {
        const auto chunk = sb::read_fvecs(base_path, kLoadChunk, start);
        searcher->add(chunk.data.data(), chunk.rows, start, opt.threads);
        std::printf("\r  added %zu / %zu", start + chunk.rows, shape.rows);
        std::fflush(stdout);
    }
    const double build_s = build_timer.elapsed_s();
    std::printf("\n  build time %.1f s\n", build_s);
    if (const std::string info = searcher->extra_info(); !info.empty()) {
        std::printf("  %s\n", info.c_str());
    }

    auto queries = sb::read_fvecs(folder + "/query.fvecs", opt.max_queries);
    auto truth = sb::read_ivecs(folder + "/groundtruth.ivecs", opt.max_queries);

    // Search on one fixed fast core, so results don't depend on which core the OS picks.
    cpu_set_t original;
    pthread_getaffinity_np(pthread_self(), sizeof(original), &original);
    if (opt.search_cpu >= 0) {
        pin_to_cpu(opt.search_cpu);
    }

    std::vector<Point> points;
    sb::Matrix<std::int32_t> found{queries.rows, opt.k,
                                   std::vector<std::int32_t>(queries.rows * opt.k)};
    for (std::size_t ef : kEfSweep) {
        if (ef < opt.k) {
            continue;
        }
        searcher->set_ef(ef);
        for (std::size_t q = 0; q < std::min(kWarmupQueries, queries.rows); ++q) {
            searcher->search(queries.row(q), opt.k, found.row(q));
        }

        // A laptop's speed drifts with temperature, so run several times and keep the median run.
        std::vector<std::pair<double, sb::LatencySummary>> runs;
        for (std::size_t r = 0; r < opt.repeats; ++r) {
            sb::LatencyRecorder latency;
            latency.reserve(queries.rows);
            sb::Stopwatch total;
            for (std::size_t q = 0; q < queries.rows; ++q) {
                sb::Stopwatch one;
                searcher->search(queries.row(q), opt.k, found.row(q));
                latency.add(one.elapsed_ns());
            }
            runs.emplace_back(static_cast<double>(queries.rows) / total.elapsed_s(),
                              latency.summarize());
        }
        std::sort(runs.begin(), runs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        const auto& median = runs[runs.size() / 2];

        Point p{ef, sb::recall_at_k(found, truth, opt.k), median.first, median.second};
        std::printf("  ef=%4zu  recall@%zu=%.4f  qps=%8.0f  p99=%8.1f us\n", ef, opt.k, p.recall,
                    p.qps, p.latency.p99_us);
        points.push_back(p);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(original), &original);

    std::filesystem::create_directories(opt.results_dir);
    const std::string out_path =
        opt.results_dir + "/" + opt.dataset + "-" + searcher->name() + ".json";
    std::ofstream out(out_path);
    out << "{\n  \"dataset\": \"" << opt.dataset << "\",\n  \"library\": \"" << searcher->name()
        << "\",\n  \"library_version\": \"" << searcher->version() << "\",\n  \"metric\": \""
        << strata::metric_name(metric) << "\",\n  \"n_base\": " << shape.rows
        << ",\n  \"dim\": " << shape.dim << ",\n  \"n_query\": " << queries.rows
        << ",\n  \"k\": " << opt.k << ",\n  \"M\": " << opt.M
        << ",\n  \"ef_construction\": " << opt.ef_construction << ",\n  \"build_threads\": "
        << (opt.threads ? opt.threads : std::thread::hardware_concurrency())
        << ",\n  \"repeats\": " << opt.repeats
        << ",\n  \"search_threads\": 1,\n  \"search_cpu\": " << opt.search_cpu
        << ",\n  \"simd\": \"" << strata::simd_level_name(strata::active_simd_level())
        << "\",\n  \"build_seconds\": " << build_s << ",\n  \"hardware\": {\"cpu\": \""
        << cpu_model() << "\"},\n  \"points\": [\n";
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point& p = points[i];
        out << "    {\"ef_search\": " << p.ef << ", \"recall\": " << p.recall
            << ", \"qps\": " << p.qps << ", \"p50_us\": " << p.latency.p50_us
            << ", \"p95_us\": " << p.latency.p95_us << ", \"p99_us\": " << p.latency.p99_us << "}"
            << (i + 1 < points.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("  saved %s\n", out_path.c_str());
}

Options parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) {
        args[argv[i]] = argv[i + 1];
    }
    if (argc % 2 == 0) {
        throw std::invalid_argument("every option needs a value");
    }
    Options opt;
    for (const auto& [key, value] : args) {
        if (key == "--dataset") {
            opt.dataset = value;
        } else if (key == "--lib") {
            opt.lib = value;
        } else if (key == "--data-dir") {
            opt.data_dir = value;
        } else if (key == "--results-dir") {
            opt.results_dir = value;
        } else if (key == "--M") {
            opt.M = std::stoul(value);
        } else if (key == "--ef-construction") {
            opt.ef_construction = std::stoul(value);
        } else if (key == "--k") {
            opt.k = std::stoul(value);
        } else if (key == "--threads") {
            opt.threads = std::stoul(value);
        } else if (key == "--search-cpu") {
            opt.search_cpu = std::stoi(value);
        } else if (key == "--queries") {
            opt.max_queries = std::stoul(value);
        } else if (key == "--simd") {
            opt.simd = value;
        } else if (key == "--repeats") {
            opt.repeats = std::max<std::size_t>(1, std::stoul(value));
        } else {
            throw std::invalid_argument("unknown option " + key);
        }
    }
    if (opt.dataset.empty() || (opt.lib != "strata" && opt.lib != "hnswlib" && opt.lib != "both")) {
        throw std::invalid_argument("--dataset is required; --lib is strata, hnswlib or both");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        if (!opt.simd.empty()) {
            const auto level = strata::parse_simd_level(opt.simd);
            if (!level) {
                throw std::invalid_argument("unknown --simd level '" + opt.simd + "'");
            }
            strata::set_simd_level(*level);
        }
        if (opt.lib == "strata" || opt.lib == "both") {
            run("strata", opt);
        }
        if (opt.lib == "hnswlib" || opt.lib == "both") {
            run("hnswlib", opt);
        }
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr,
                     "error: %s\nusage: %s --dataset NAME [--lib strata|hnswlib|both] [--M 16]\n"
                     "       [--ef-construction 200] [--k 10] [--threads N] [--search-cpu 2]\n"
                     "       [--queries N] [--simd scalar|autovec|avx2|avx512]\n"
                     "       [--repeats 3]\n",
                     e.what(), argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
