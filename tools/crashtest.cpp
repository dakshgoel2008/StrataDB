// Kill-9 crash test.
//
// The parent starts a child process that inserts vectors from several threads
// and takes checkpoints. After each insert returns, the child reports the row id
// to the parent through a pipe. At a random moment the parent kills the child
// with SIGKILL, opens the database itself, and checks that:
//   - every row the child reported is there, with exactly the right vector;
//   - every other row is a complete vector, never garbage.
// Then it starts the next child on the same database.
//
// Usage: crashtest [--iterations 1000] [--dir crashtest-runs/db] [--threads 4]
//                  [--sync group|every-write|none] [--max-run-ms 50]
//                  [--reset-every 200] [--seed 1]

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "strata/database.h"

namespace {

constexpr std::size_t kDim = 16;
constexpr std::size_t kCapacity = 200'000;

struct Options {
    std::size_t iterations = 1000;
    std::string dir = "crashtest-runs/db";
    std::size_t threads = 4;
    std::string sync = "group";
    int max_run_ms = 50;
    std::size_t reset_every = 200;
    std::uint64_t seed = 1;
};

// What a child reports after an insert returns. 16 bytes, so writes to the
// pipe from different threads never mix (pipes keep small writes whole).
struct Ack {
    std::uint64_t row;
    std::uint32_t thread;
    std::uint32_t seq;
};

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

float checksum(const float* v) {
    float sum = 0;
    for (std::size_t i = 0; i + 1 < kDim; ++i) {
        sum += v[i] * static_cast<float>(i + 1);
    }
    return sum;
}

// The vector a child inserts as its thread's seq-th insert. The last value is a
// checksum of the others, so a half-written or garbage vector is easy to spot.
std::vector<float> make_vector(std::uint64_t child_seed, std::uint32_t thread, std::uint32_t seq) {
    std::vector<float> v(kDim);
    std::uint64_t state = mix(child_seed ^ (std::uint64_t{thread} << 32) ^ seq);
    for (std::size_t i = 0; i + 1 < kDim; ++i) {
        state = mix(state);
        v[i] = static_cast<float>(state >> 40) / static_cast<float>(1 << 24);
    }
    v[kDim - 1] = checksum(v.data());
    return v;
}

strata::DatabaseOptions db_options(const Options& opt) {
    strata::DatabaseOptions o;
    o.sync = strata::parse_sync_mode(opt.sync);
    o.wal_file_bytes = 1u << 20;  // small, so new log files are started often
    return o;
}

int run_child(const Options& opt, std::uint64_t child_seed, int fd) {
    try {
        auto db = strata::Database::open(opt.dir, db_options(opt));
        std::vector<std::jthread> threads;
        threads.reserve(opt.threads);
        for (std::size_t t = 0; t < opt.threads; ++t) {
            threads.emplace_back([&, t] {
                const auto thread = static_cast<std::uint32_t>(t);
                for (std::uint32_t seq = 0;; ++seq) {
                    const auto v = make_vector(child_seed, thread, seq);
                    const Ack ack{static_cast<std::uint64_t>(db->insert(v)), thread, seq};
                    if (::write(fd, &ack, sizeof(ack)) != static_cast<ssize_t>(sizeof(ack))) {
                        std::_Exit(4);
                    }
                }
            });
        }
        // Checkpoints at random moments, so some kills land in the middle of one.
        std::mt19937_64 rng(child_seed);
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(rng() % 20));
            db->checkpoint();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "child error: %s\n", e.what());
        std::_Exit(3);
    }
}

struct Expected {
    std::uint64_t child_seed;
    std::uint32_t thread;
    std::uint32_t seq;
};

struct Totals {
    std::size_t acked = 0;
    std::size_t rows_checked = 0;
    std::size_t dropped_tails = 0;
    std::size_t replayed_records = 0;
    std::size_t search_misses = 0;
    double recovery_seconds = 0;
};

// Opens the database, as after a real crash, and checks it. Returns an error
// message, or an empty string if everything is right.
std::string verify(const Options& opt, const std::unordered_map<std::uint64_t, Expected>& acked,
                   std::uint64_t max_acked, Totals& totals) {
    const auto start = std::chrono::steady_clock::now();
    auto db = strata::Database::open(opt.dir, db_options(opt));
    totals.recovery_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (db->recovery().dropped_bytes > 0) {
        ++totals.dropped_tails;
    }
    totals.replayed_records += db->recovery().replayed_records;

    const std::size_t size = db->size();
    if (!acked.empty() && size <= max_acked) {
        return "row " + std::to_string(max_acked) + " was acknowledged but only " +
               std::to_string(size) + " rows came back";
    }
    std::vector<float> got(kDim);
    for (std::size_t row = 0; row < size; ++row) {
        db->index().get_vector(static_cast<strata::RowId>(row), got);
        if (got[kDim - 1] != checksum(got.data())) {
            return "row " + std::to_string(row) + " holds a damaged vector";
        }
        const auto it = acked.find(row);
        if (it != acked.end() &&
            got != make_vector(it->second.child_seed, it->second.thread, it->second.seq)) {
            return "row " + std::to_string(row) + " holds the wrong vector";
        }
    }
    totals.rows_checked += size;

    // Search must work on the recovered graph. HNSW is approximate, so a miss is
    // counted, not treated as an error.
    if (size > 0) {
        db->index().get_vector(static_cast<strata::RowId>(size / 2), got);
        std::vector<strata::RowId> ids(1);
        std::vector<float> dist(1);
        db->search(got, 1, 64, ids, dist);
        if (ids[0] < 0 || static_cast<std::size_t>(ids[0]) >= size) {
            return "search returned a row that doesn't exist";
        }
        if (dist[0] != 0.0f) {
            ++totals.search_misses;
        }
    }
    return {};
}

int run_parent(const Options& opt, const char* self) {
    std::mt19937_64 rng(opt.seed);
    std::unordered_map<std::uint64_t, Expected> acked;  // row -> what the child inserted there
    std::uint64_t max_acked = 0;
    Totals totals;

    for (std::size_t it = 0; it < opt.iterations; ++it) {
        if (it % opt.reset_every == 0) {
            std::filesystem::remove_all(opt.dir);
            strata::Database::create(
                opt.dir,
                std::make_unique<strata::HnswIndex>(kDim, strata::Metric::L2, kCapacity,
                                                    strata::HnswParams{8, 40, 1}),
                db_options(opt));
            acked.clear();
            max_acked = 0;
        }

        int fds[2];
        if (::pipe(fds) != 0) {
            std::perror("pipe");
            return 1;
        }
        const std::uint64_t child_seed = rng();
        const pid_t pid = ::fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            ::close(fds[0]);
            const std::string seed = std::to_string(child_seed);
            const std::string fd = std::to_string(fds[1]);
            const std::string threads = std::to_string(opt.threads);
            ::execl(self, self, "--child", opt.dir.c_str(), seed.c_str(), fd.c_str(),
                    threads.c_str(), opt.sync.c_str(), static_cast<char*>(nullptr));
            std::_Exit(5);
        }
        ::close(fds[1]);

        // Collect reports until the random kill time, then kill and collect the rest.
        std::vector<char> data;
        char buffer[1 << 16];
        const auto run_ms =
            static_cast<int>(rng() % static_cast<std::uint64_t>(opt.max_run_ms)) + 1;
        const auto kill_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(run_ms);
        bool killed = false;
        while (true) {
            if (!killed) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      kill_at - std::chrono::steady_clock::now())
                                      .count();
                if (left <= 0) {
                    ::kill(pid, SIGKILL);
                    killed = true;
                    continue;
                }
                pollfd p{fds[0], POLLIN, 0};
                ::poll(&p, 1, static_cast<int>(left));
                if ((p.revents & (POLLIN | POLLHUP)) == 0) {
                    continue;
                }
            }
            const ssize_t n = ::read(fds[0], buffer, sizeof(buffer));
            if (n > 0) {
                data.insert(data.end(), buffer, buffer + n);
            } else if (n == 0) {
                if (killed) {
                    break;
                }
                ::kill(pid, SIGKILL);  // the child exited on its own; that's an error below
                killed = true;
            }
        }
        ::close(fds[0]);
        int status = 0;
        ::waitpid(pid, &status, 0);
        if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGKILL) {
            std::fprintf(stderr, "iteration %zu: child stopped by itself (status %d)\n", it,
                         status);
            return 1;
        }

        for (std::size_t off = 0; off + sizeof(Ack) <= data.size(); off += sizeof(Ack)) {
            Ack ack;
            std::memcpy(&ack, data.data() + off, sizeof(Ack));
            acked[ack.row] = {child_seed, ack.thread, ack.seq};
            max_acked = std::max(max_acked, ack.row);
            ++totals.acked;
        }

        try {
            const std::string error = verify(opt, acked, max_acked, totals);
            if (!error.empty()) {
                std::fprintf(stderr, "iteration %zu: %s\n", it, error.c_str());
                return 1;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "iteration %zu: recovery failed: %s\n", it, e.what());
            return 1;
        }
        if ((it + 1) % 100 == 0) {
            std::printf("%zu iterations, %zu acknowledged inserts checked\n", it + 1, totals.acked);
            std::fflush(stdout);
        }
    }

    std::printf("\nPASSED: %zu kill-9 crashes (sync mode %s)\n", opt.iterations, opt.sync.c_str());
    std::printf("  acknowledged inserts checked: %zu, none lost\n", totals.acked);
    std::printf("  rows checked for damage:      %zu, none damaged\n", totals.rows_checked);
    std::printf("  crashes that left a cut-off log record: %zu\n", totals.dropped_tails);
    std::printf("  log records replayed: %zu, average recovery %.1f ms\n", totals.replayed_records,
                1000.0 * totals.recovery_seconds / static_cast<double>(opt.iterations));
    std::printf("  searches that missed their stored vector: %zu of %zu\n", totals.search_misses,
                opt.iterations);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        if (argc == 7 && std::string(argv[1]) == "--child") {
            opt.dir = argv[2];
            opt.threads = std::stoul(argv[5]);
            opt.sync = argv[6];
            return run_child(opt, std::stoull(argv[3]), std::stoi(argv[4]));
        }
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(arg + " needs a value");
                }
                return argv[++i];
            };
            if (arg == "--iterations") {
                opt.iterations = std::stoul(value());
            } else if (arg == "--dir") {
                opt.dir = value();
            } else if (arg == "--threads") {
                opt.threads = std::stoul(value());
            } else if (arg == "--sync") {
                opt.sync = value();
                strata::parse_sync_mode(opt.sync);
            } else if (arg == "--max-run-ms") {
                opt.max_run_ms = std::stoi(value());
            } else if (arg == "--reset-every") {
                opt.reset_every = std::stoul(value());
            } else if (arg == "--seed") {
                opt.seed = std::stoull(value());
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }
        if (opt.threads == 0 || opt.max_run_ms <= 0 || opt.reset_every == 0) {
            throw std::invalid_argument("threads, max-run-ms and reset-every must be positive");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    char self[4096];
    const ssize_t len = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (len <= 0) {
        std::perror("readlink");
        return 1;
    }
    self[len] = '\0';
    return run_parent(opt, self);
}
