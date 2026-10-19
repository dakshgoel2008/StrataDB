#!/usr/bin/env bash
# Reproduces every number in docs/results.md on a fresh clone.
#
# Usage:
#   bench/run_all.sh              everything (several hours, needs ~15 GB of disk for data)
#   bench/run_all.sh sift         only the SIFT parts (about 1 hour)
#
# Needs: cmake, ninja, a C++20 compiler, python3. For the fuzzers, clang++.
# Results go to bench/results/, plots to bench/results/plots/.

set -euo pipefail
cd "$(dirname "$0")/.."

ONLY="${1:-all}"
SEARCH_CPU="${SEARCH_CPU:-2}"  # a fast core; see bench/hardware.md

step() { printf '\n==== %s ====\n' "$1"; }

step "Python packages"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install -q -r bench/requirements.txt
PY=.venv/bin/python

step "Build and test"
cmake --preset release
cmake --build --preset release
ctest --preset release

step "Datasets"
$PY bench/prepare_datasets.py sift
if [ "$ONLY" = "all" ]; then
    $PY bench/prepare_datasets.py cohere-1m
fi

step "Exact search matches FAISS"
./build/release/tools/compute_groundtruth --base data/sift/base.fvecs \
    --query data/sift/query.fvecs --metric l2 --k 100 --out data/sift/groundtruth_strata
$PY bench/check_groundtruth.py sift

step "Distance kernels"
taskset -c "$SEARCH_CPU" ./build/release/bench/distance_bench --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true

step "HNSW on SIFT-1M: StrataDB vs hnswlib"
./build/release/bench/hnsw_bench --dataset sift --lib both --simd avx2 --repeats 5 \
    --search-cpu "$SEARCH_CPU"
./build/release/bench/hnsw_bench --dataset sift --lib strata --simd scalar --repeats 5 \
    --search-cpu "$SEARCH_CPU"
$PY bench/run_baselines.py sift --libs faiss
$PY bench/plot_baselines.py sift

if [ "$ONLY" = "all" ]; then
    step "Cohere-1M: full floats, SQ8 and PQ with re-ranking"
    BENCH="./build/release/bench/hnsw_bench --dataset cohere-1m --search-cpu $SEARCH_CPU"
    $BENCH --lib both
    $BENCH --lib strata --encoding sq8 --rerank 0,20,50,100
    $BENCH --lib strata --encoding pq --pq-m 96 --rerank 0,50,100,200
    $BENCH --lib strata --encoding pq --pq-m 48 --rerank 0,100,200,400
    $PY bench/plot_baselines.py cohere-1m
    $PY bench/plot_rerank.py cohere-1m
fi

step "Durability: insert speed and recovery time"
./build/release/bench/durability_bench insert
./build/release/bench/durability_bench recovery
$PY bench/plot_durability.py sift

step "Kill-9 crash test"
./build/release/tools/crashtest --iterations 10000 --sync group --dir crashtest-runs/group
./build/release/tools/crashtest --iterations 3000 --sync every-write --dir crashtest-runs/every
./build/release/tools/crashtest --iterations 3000 --sync none --dir crashtest-runs/none

if command -v clang++ >/dev/null; then
    step "Fuzzing, 1 hour each"
    cmake --preset fuzz
    cmake --build --preset fuzz --target wal_fuzz snapshot_fuzz make_seeds
    ./build/fuzz/fuzz/make_seeds fuzz/corpus
    mkdir -p fuzz/artifacts
    for target in wal snapshot; do
        ./build/fuzz/fuzz/${target}_fuzz "fuzz/corpus/$target" -max_total_time=3600 \
            -artifact_prefix=fuzz/artifacts/ -print_final_stats=1
    done
else
    step "Fuzzing skipped (clang++ not found)"
fi

step "Done. Results are in bench/results/, plots in bench/results/plots/"
