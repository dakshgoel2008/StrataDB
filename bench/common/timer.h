#pragma once

#include <chrono>
#include <cstdint>

namespace strata::bench {

// Measures elapsed time. Uses steady_clock because it never jumps when
// the system clock is adjusted.
class Stopwatch {
public:
    Stopwatch() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    std::uint64_t elapsed_ns() const {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_);
        return static_cast<std::uint64_t>(ns.count());
    }

    double elapsed_s() const { return static_cast<double>(elapsed_ns()) / 1e9; }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

}  // namespace strata::bench
