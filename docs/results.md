# Results

Hardware: [bench/hardware.md](../bench/hardware.md) (Intel i5-1235U, AVX2). Search is
single-threaded, pinned to CPU 2. Reproduce with `bench/run_all.sh`.

## Exact search

`compute_groundtruth` vs FAISS `IndexFlat` and the datasets' published answers
(`bench/check_groundtruth.py`).

| Dataset | Top-k | Same as FAISS | Same as published answers |
|---|---|---|---|
| SIFT-1M (L2) | 100 | 100% of 10,000 queries, same order | 98.88%; all differences are ties |
| Cohere-1M (cosine) | 10 | 100% of 1,000 queries | 100% |

A tie is a vector at exactly the k-th distance; either answer is correct.

## Distance kernels

`taskset -c 2 ./build/release/bench/distance_bench --benchmark_repetitions=3`

Nanoseconds per distance, median of 3.

| Kernel | dim | scalar | autovec | AVX2 | AVX2 vs scalar |
|---|---|---|---|---|---|
| L2 | 128 | 11.2 | 6.4 | 5.6 | 2.0x |
| L2 | 768 | 76.9 | 39.9 | 26.8 | 2.9x |
| L2 | 960 | 99.1 | 50.0 | 32.8 | 3.0x |
| dot | 128 | 10.0 | 7.9 | 5.0 | 2.0x |
| dot | 768 | 71.0 | 33.6 | 22.7 | 3.1x |
| dot | 960 | 91.6 | 43.5 | 28.3 | 3.2x |

- autovec: plain loop compiled with `-mavx2 -mfma -ffast-math`.
- AVX-512: verified with Intel SDE; not benchmarked (no AVX-512 on this CPU).

## HNSW search, SIFT-1M

`./build/release/bench/hnsw_bench --dataset sift --lib both --simd avx2 --repeats 5`

1M x 128, M = 16, ef_construction = 200, k = 10, 10,000 queries. Queries/s is the
median of 5 runs. hnswlib 0.8.0 is compiled into the same binary with `-march=native`.

| ef_search | recall@10 | StrataDB scalar | StrataDB AVX2 | hnswlib |
|---|---|---|---|---|
| 32 | 0.903 | 11,499 | 17,676 | 14,844 |
| 48 | 0.943 | 8,954 | 12,807 | 10,582 |
| 64 | 0.963 | 7,428 | 10,319 | 9,637 |
| 96 | 0.981 | 5,213 | 7,190 | 6,761 |
| 128 | 0.989 | 4,229 | 4,921 | 4,832 |

- p99 latency at ef 48: StrataDB AVX2 121 us, hnswlib 167 us.
- Build time, 12 threads: StrataDB 102 s, hnswlib 114 s.
- AVX2 vs scalar, end to end at ef 48: 1.43x.
- Recall equals hnswlib at every ef_search.
- Run-to-run variation is up to 25% (thermal); differences under 15% are noise.

## Compression and re-ranking, Cohere-1M

`./build/release/bench/hnsw_bench --dataset cohere-1m --encoding sq8|pq --rerank ...`

1M x 768, cosine, M = 16, ef_construction = 200, k = 10, 1,000 queries, median of 3.
Quantizers trained on 50,000 vectors. Re-ranking reads full vectors from the
memory-mapped dataset file (not counted in index memory).

| Index | Bytes per vector | Index memory | Build (s) |
|---|---|---|---|
| Full floats | 3,072 | 3,121 MiB | 438 |
| SQ8 | 768 (4x) | 924 MiB | 269 |
| PQ, 96 pieces | 96 (32x) | 283 MiB | 349 |
| PQ, 48 pieces | 48 (64x) | 237 MiB | 261 |

Index memory includes the uncompressed graph links.

| Index | Re-rank depth | ef_search | recall@10 | Queries/s | p99 (us) |
|---|---|---|---|---|---|
| hnswlib, full floats | - | 96 | 0.961 | 2,144 | 643 |
| StrataDB, full floats | - | 96 | 0.961 | 2,427 | 594 |
| StrataDB, SQ8 | none | 96 | 0.954 | 2,786 | 550 |
| StrataDB, SQ8 | 20 | 96 | 0.961 | 2,955 | 494 |
| StrataDB, PQ 96 | none | 512 | 0.575 | 1,207 | 1,026 |
| StrataDB, PQ 96 | 100 | 512 | 0.937 | 1,272 | 975 |
| StrataDB, PQ 96 | 200 | 384 | 0.955 | 1,495 | 829 |
| StrataDB, PQ 48 | 400 | 512 | 0.857 | 1,238 | 945 |

- SQ8 is faster than full floats: each distance reads 4x less memory.
- PQ without re-ranking plateaus near 0.57 recall at any ef_search.
- PQ 48 (64x) does not reach 0.95 even at re-rank depth 400.

## Durability, SIFT

`./build/release/bench/durability_bench insert` and `durability_bench recovery`

128 dims, M = 16, ef_construction = 200, NVMe SSD (ext4). "Database" is the full
`Database::insert` (log + HNSW insert); "log only" is the write-ahead log alone.

### Insert throughput

| Sync mode | Threads | Log only (inserts/s) | Database (inserts/s) | Inserts per fdatasync | Database p99 (us) |
|---|---|---|---|---|---|
| every-write | 1 | 1,549 | 1,140 | 1.0 | 1,394 |
| every-write | 16 | 1,415 | 1,221 | 1.0 | 21,139 |
| group | 1 | 1,520 | 1,140 | 1.0 | 1,337 |
| group | 4 | 3,781 | 2,962 | 2.0 | 1,965 |
| group | 8 | 6,513 | 5,989 | 4.0 | 1,811 |
| group | 16 | 11,285 | 9,702 | 7.9 | 2,668 |
| none | 1 | 653,618 | 9,335 | - | 228 |
| none | 16 | 305,379 | 26,779 | - | 3,436 |

- One fdatasync takes about 0.65 ms, capping every-write near 1,500/s.
- Group commit, 16 threads: 7.9x every-write, same durability guarantee.

### Recovery time

Snapshot of 800,000 vectors (496 MiB, written in 0.62 s) plus N log records.

| Log records since checkpoint | Snapshot load (s) | Replay, 1 thread (s) | Replay, 12 threads (s) |
|---|---|---|---|
| 0 | 0.38 | 0.00 | 0.00 |
| 20,000 | 0.40 | 5.39 | 2.36 |
| 50,000 | 0.40 | 14.05 | 5.83 |
| 100,000 | 0.42 | 27.58 | 11.92 |
| 200,000 | 0.49 | 60.59 | 24.61 |

- Replay costs about 0.12 ms per record (12 threads); each record is an HNSW insert.
- Snapshot load measured with the file in the page cache.

### Crash testing and fuzzing

| Test | Result |
|---|---|
| Kill-9, group commit | 10,000 crashes, 188,462 acknowledged inserts checked, 0 lost or damaged |
| Kill-9, every-write | 3,000 crashes, 19,124 acknowledged inserts checked, 0 lost or damaged |
| Kill-9, no sync | 3,000 crashes, 139,621 acknowledged inserts checked, 0 lost or damaged |
| libFuzzer + ASan, log reader and recovery | 1 hour, 18.6M inputs, 0 crashes |
| libFuzzer + ASan, snapshot loader | 1 hour, 46.1M inputs, 0 crashes |
