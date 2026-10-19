// Measures what durability costs.
//
//   durability_bench insert    insert throughput and latency for each sync mode and
//                              thread count, for the whole database and the log alone
//   durability_bench recovery  time to open a database after a crash, for different
//                              numbers of log records since the last checkpoint
//
// Common:   --data data --dataset sift --dir data/bench-db
// insert:   --modes every-write,group,none --threads 1,4,8,16 --seconds 3 --pool 300000
// recovery: --snapshot-rows 800000 --tails 0,20000,50000,100000,200000

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bench/common/metrics.h"
#include "bench/common/timer.h"
#include "bench/common/vecs_io.h"
#include "strata/database.h"
#include "strata/hnsw_index.h"
#include "strata/wal.h"

namespace sb = strata::bench;

namespace {

struct Options {
    std::string command;
    std::string data_dir = "data";
    std::string dataset = "sift";
    std::string dir = "data/bench-db";
    std::string results_dir = "bench/results/durability";
    std::vector<std::string> modes = {"every-write", "group", "none"};
    std::vector<std::size_t> threads = {1, 4, 8, 16};
    double seconds = 3;
    std::size_t pool = 300000;
    std::size_t snapshot_rows = 800000;
    std::vector<std::size_t> tails = {0, 20000, 50000, 100000, 200000};
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) {
        out.push_back(item);
    }
    return out;
}

std::vector<std::size_t> split_numbers(const std::string& s) {
    std::vector<std::size_t> out;
    for (const auto& item : split(s)) {
        out.push_back(std::stoul(item));
    }
    return out;
}

const strata::HnswParams kParams{16, 200, 100};

struct InsertResult {
    std::string target;
    std::string mode;
    std::size_t threads = 0;
    std::size_t inserts = 0;
    double seconds = 0;
    std::uint64_t syncs = 0;
    sb::LatencySummary latency;
};

// Runs `threads` threads that each call insert_one(row) for new rows until the
// time is up or the rows run out.
template <typename Fn>
InsertResult run_inserts(std::size_t threads, double seconds, std::size_t rows, Fn insert_one) {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::vector<std::vector<std::uint64_t>> times(threads);
    const sb::Stopwatch total;
    {
        std::vector<std::jthread> workers;
        workers.reserve(threads);
        for (std::size_t t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                while (!stop.load(std::memory_order_relaxed)) {
                    const std::size_t row = next.fetch_add(1);
                    if (row >= rows) {
                        return;
                    }
                    const sb::Stopwatch one;
                    insert_one(row);
                    times[t].push_back(one.elapsed_ns());
                }
            });
        }
        while (total.elapsed_s() < seconds && next.load() < rows) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        stop = true;
    }
    InsertResult r;
    r.seconds = total.elapsed_s();
    r.threads = threads;
    sb::LatencyRecorder all;
    for (const auto& list : times) {
        for (std::uint64_t ns : list) {
            all.add(ns);
        }
    }
    r.inserts = all.count();
    if (r.inserts > 0) {
        r.latency = all.summarize();
    }
    return r;
}

void insert_command(const Options& opt) {
    const auto base = sb::read_fvecs(opt.data_dir + "/" + opt.dataset + "/base.fvecs", opt.pool);
    const std::size_t dim = base.dim;
    std::printf("%zu vectors of %zu dims, %.0f s per run\n\n", base.rows, dim, opt.seconds);
    std::printf("%-9s %-12s %7s %11s %8s %9s %10s %10s\n", "target", "sync mode", "threads",
                "inserts/s", "syncs/s", "per sync", "p50 (us)", "p99 (us)");

    std::vector<InsertResult> results;
    for (const std::string& mode_name : opt.modes) {
        const strata::SyncMode mode = strata::parse_sync_mode(mode_name);
        for (std::size_t threads : opt.threads) {
            for (const bool log_only : {true, false}) {
                std::filesystem::remove_all(opt.dir);
                std::filesystem::create_directories(opt.dir);
                InsertResult r;
                if (log_only) {
                    strata::WriteAheadLog wal(strata::local_file_system(), opt.dir, 1,
                                              {mode, 64u << 20}, false);
                    r = run_inserts(threads, opt.seconds, base.rows, [&](std::size_t row) {
                        const auto payload =
                            strata::encode_insert(row, std::span(base.row(row), dim), dim);
                        wal.wait(wal.append(strata::WalRecordType::Insert, payload));
                    });
                    r.syncs = wal.sync_count();
                    r.target = "log only";
                } else {
                    strata::DatabaseOptions o;
                    o.sync = mode;
                    auto db =
                        strata::Database::create(opt.dir,
                                                 std::make_unique<strata::HnswIndex>(
                                                     dim, strata::Metric::L2, base.rows, kParams),
                                                 o);
                    r = run_inserts(threads, opt.seconds, base.rows, [&](std::size_t row) {
                        db->insert(std::span(base.row(row), dim));
                    });
                    r.syncs = db->wal().sync_count();
                    r.target = "database";
                }
                r.mode = mode_name;
                results.push_back(r);
                const double per_sync =
                    r.syncs > 0 ? static_cast<double>(r.inserts) / static_cast<double>(r.syncs)
                                : 0.0;
                std::printf("%-9s %-12s %7zu %11.0f %8.0f %9.1f %10.1f %10.1f\n", r.target.c_str(),
                            r.mode.c_str(), threads, static_cast<double>(r.inserts) / r.seconds,
                            static_cast<double>(r.syncs) / r.seconds, per_sync, r.latency.p50_us,
                            r.latency.p99_us);
                std::fflush(stdout);
            }
        }
    }
    std::filesystem::remove_all(opt.dir);

    std::filesystem::create_directories(opt.results_dir);
    const std::string path = opt.results_dir + "/insert-" + opt.dataset + ".json";
    std::ofstream out(path);
    out << "{\n  \"dataset\": \"" << opt.dataset << "\",\n  \"dim\": " << dim
        << ",\n  \"seconds_per_run\": " << opt.seconds << ",\n  \"runs\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\"target\": \"" << r.target << "\", \"sync\": \"" << r.mode
            << "\", \"threads\": " << r.threads << ", \"inserts\": " << r.inserts
            << ", \"seconds\": " << r.seconds << ", \"syncs\": " << r.syncs
            << ", \"p50_us\": " << r.latency.p50_us << ", \"p99_us\": " << r.latency.p99_us << "}"
            << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nsaved %s\n", path.c_str());
}

struct RecoveryPoint {
    std::size_t tail = 0;
    std::size_t replay_threads = 0;
    double load_seconds = 0;
    double replay_seconds = 0;
};

void recovery_command(const Options& opt) {
    const std::size_t max_tail = *std::max_element(opt.tails.begin(), opt.tails.end());
    const auto base = sb::read_fvecs(opt.data_dir + "/" + opt.dataset + "/base.fvecs",
                                     opt.snapshot_rows + max_tail);
    if (base.rows < opt.snapshot_rows + max_tail) {
        throw std::runtime_error("dataset has fewer rows than snapshot-rows + the largest tail");
    }
    const std::size_t dim = base.dim;
    const std::size_t capacity = opt.snapshot_rows + max_tail;
    const std::string seed_dir = opt.dir + "/snapshot-only";

    // One database whose snapshot holds snapshot_rows vectors and whose log is empty.
    std::filesystem::remove_all(opt.dir);
    std::printf("building an index of %zu vectors...\n", opt.snapshot_rows);
    sb::Stopwatch watch;
    auto index = std::make_unique<strata::HnswIndex>(dim, strata::Metric::L2, capacity, kParams);
    index->add(std::span(base.data).first(opt.snapshot_rows * dim), 0);
    std::printf("  built in %.1f s\n", watch.elapsed_s());
    watch.reset();
    strata::DatabaseOptions o;
    o.sync = strata::SyncMode::None;
    strata::Database::create(seed_dir, std::move(index), o);
    const double snapshot_write_s = watch.elapsed_s();
    std::uintmax_t snapshot_bytes = 0;
    for (const auto& f : std::filesystem::directory_iterator(seed_dir)) {
        if (f.path().extension() == ".snap") {
            snapshot_bytes = f.file_size();
        }
    }
    std::printf("  snapshot written in %.2f s, %.1f MiB\n\n", snapshot_write_s,
                static_cast<double>(snapshot_bytes) / (1 << 20));

    std::printf("%12s %8s %10s %11s %10s\n", "log records", "threads", "load (s)", "replay (s)",
                "total (s)");
    std::vector<RecoveryPoint> points;
    for (std::size_t tail : opt.tails) {
        const std::string dir = opt.dir + "/tail-" + std::to_string(tail);
        std::filesystem::copy(seed_dir, dir, std::filesystem::copy_options::recursive);
        {
            // One record per insert, as online writes would make. Many threads keep it quick.
            auto db = strata::Database::open(dir, o);
            run_inserts(std::thread::hardware_concurrency(), 1e9, tail, [&](std::size_t i) {
                db->insert(std::span(base.row(opt.snapshot_rows + i), dim));
            });
        }
        for (std::size_t threads : {std::size_t{1}, std::size_t{0}}) {
            strata::DatabaseOptions ro;
            ro.replay_threads = threads;
            const auto db = strata::Database::open(dir, ro);
            RecoveryPoint p;
            p.tail = tail;
            p.replay_threads = threads == 0 ? std::thread::hardware_concurrency() : 1;
            p.load_seconds = db->recovery().load_seconds;
            p.replay_seconds = db->recovery().replay_seconds;
            points.push_back(p);
            std::printf("%12zu %8zu %10.2f %11.2f %10.2f\n", tail, p.replay_threads, p.load_seconds,
                        p.replay_seconds, p.load_seconds + p.replay_seconds);
            std::fflush(stdout);
        }
        std::filesystem::remove_all(dir);
    }
    std::filesystem::remove_all(opt.dir);

    std::filesystem::create_directories(opt.results_dir);
    const std::string path = opt.results_dir + "/recovery-" + opt.dataset + ".json";
    std::ofstream out(path);
    out << "{\n  \"dataset\": \"" << opt.dataset
        << "\",\n  \"snapshot_rows\": " << opt.snapshot_rows
        << ",\n  \"snapshot_bytes\": " << snapshot_bytes
        << ",\n  \"snapshot_write_seconds\": " << snapshot_write_s << ",\n  \"points\": [\n";
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        out << "    {\"log_records\": " << p.tail << ", \"replay_threads\": " << p.replay_threads
            << ", \"load_seconds\": " << p.load_seconds
            << ", \"replay_seconds\": " << p.replay_seconds << "}"
            << (i + 1 < points.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nsaved %s\n", path.c_str());
}

Options parse(int argc, char** argv) {
    Options opt;
    if (argc < 2) {
        throw std::invalid_argument("usage: durability_bench insert|recovery [options]");
    }
    opt.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument(arg + " needs a value");
        }
        const std::string value = argv[++i];
        if (arg == "--data") {
            opt.data_dir = value;
        } else if (arg == "--dataset") {
            opt.dataset = value;
        } else if (arg == "--dir") {
            opt.dir = value;
        } else if (arg == "--modes") {
            opt.modes = split(value);
        } else if (arg == "--threads") {
            opt.threads = split_numbers(value);
        } else if (arg == "--seconds") {
            opt.seconds = std::stod(value);
        } else if (arg == "--pool") {
            opt.pool = std::stoul(value);
        } else if (arg == "--snapshot-rows") {
            opt.snapshot_rows = std::stoul(value);
        } else if (arg == "--tails") {
            opt.tails = split_numbers(value);
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        if (opt.command == "insert") {
            insert_command(opt);
        } else if (opt.command == "recovery") {
            recovery_command(opt);
        } else {
            throw std::invalid_argument("unknown command " + opt.command);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
