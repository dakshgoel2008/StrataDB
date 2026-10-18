// Speed of each distance kernel at common vector sizes, one vector pair that
// stays in the CPU cache. Run it pinned to a fast core:
//   taskset -c 2 ./build/release/bench/distance_bench

#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "strata/distance.h"

namespace {

void run_kernel(benchmark::State& state, strata::KernelFn fn) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(dim), b(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(fn(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));  // floats per second
}

void register_all() {
    for (auto level : strata::supported_simd_levels()) {
        const auto k = strata::kernels_for(level);
        const std::string name(strata::simd_level_name(level));
        for (auto [label, fn] : {std::pair{"l2", k.l2_squared}, std::pair{"dot", k.dot_product}}) {
            benchmark::RegisterBenchmark(std::string(label) + "/" + name,
                                         [fn](benchmark::State& s) { run_kernel(s, fn); })
                ->Arg(128)
                ->Arg(768)
                ->Arg(960);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    register_all();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
