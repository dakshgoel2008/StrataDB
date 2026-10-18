#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace strata {

// Calls fn(i) for every i in [0, n), using up to `threads` threads (0 = all cores).
// Threads take small batches from a shared counter, so fast and slow cores
// finish at about the same time. The calling thread does work too.
template <typename Fn>
void parallel_for(std::size_t n, std::size_t threads, Fn&& fn, std::size_t batch = 8) {
    if (n == 0) {
        return;
    }
    if (threads == 0) {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }
    threads = std::min(threads, (n + batch - 1) / batch);

    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        while (true) {
            const std::size_t start = next.fetch_add(batch);
            if (start >= n) {
                return;
            }
            const std::size_t end = std::min(start + batch, n);
            for (std::size_t i = start; i < end; ++i) {
                fn(i);
            }
        }
    };

    std::vector<std::jthread> pool;
    pool.reserve(threads - 1);
    for (std::size_t t = 1; t < threads; ++t) {
        pool.emplace_back(worker);
    }
    worker();
}

}  // namespace strata
