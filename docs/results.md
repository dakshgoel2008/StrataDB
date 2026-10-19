# Results

Measured on the laptop described in [bench/hardware.md](../bench/hardware.md):
Intel i5-1235U, AVX2, single-thread search pinned to CPU 2 (a fast core).

## Distance kernels

`taskset -c 2 ./build/release/bench/distance_bench --benchmark_repetitions=3`

Time per distance in nanoseconds (median of 3, one vector pair in cache).

| Kernel | dim | scalar | autovec | AVX2 | AVX2 vs scalar |
|---|---|---|---|---|---|
| L2 | 128 | 11.2 | 6.4 | 5.6 | 2.0x |
| L2 | 768 | 76.9 | 39.9 | 26.8 | 2.9x |
| L2 | 960 | 99.1 | 50.0 | 32.8 | 3.0x |
| dot | 128 | 10.0 | 7.9 | 5.0 | 2.0x |
| dot | 768 | 71.0 | 33.6 | 22.7 | 3.1x |
| dot | 960 | 91.6 | 43.5 | 28.3 | 3.2x |

- **scalar**: plain loops with four running sums, built for any x86-64 CPU.
- **autovec**: a plain loop built with `-mavx2 -mfma -ffast-math`, so the compiler
  vectorizes it itself. At 128 dimensions it gets most of the gain; hand-written AVX2
  pulls ahead at 768 and 960.
- **AVX-512** kernels are checked for correctness under Intel SDE (this CPU has no
  AVX-512) and are not benchmarked here.

## HNSW search, SIFT-1M

`./build/release/bench/hnsw_bench --dataset sift --lib both --simd avx2 --repeats 5`

1M vectors, 128 dimensions, M = 16, ef_construction = 200, k = 10, 10,000 queries.
Queries per second is the median of 5 runs. hnswlib 0.8.0 is compiled into the same
program with `-march=native`.

| ef_search | recall@10 | StrataDB scalar | StrataDB AVX2 | hnswlib |
|---|---|---|---|---|
| 32 | 0.903 | 11,499 | 17,676 | 14,844 |
| 48 | 0.943 | 8,954 | 12,807 | 10,582 |
| 64 | 0.963 | 7,428 | 10,319 | 9,637 |
| 96 | 0.981 | 5,213 | 7,190 | 6,761 |
| 128 | 0.989 | 4,229 | 4,921 | 4,832 |

p99 latency at recall 0.943 (ef 48): StrataDB AVX2 121 us, hnswlib 167 us.

Build time with 12 threads: StrataDB AVX2 102 s, hnswlib 114 s.

### Notes

- Recall is the same as hnswlib at every ef_search, so the graphs are of equal quality.
- Laptop speed changes with temperature. Single runs of the same setting differed by
  up to 25%, which is why every point is the median of 5 runs. Treat differences under
  about 15% as noise.
- Prefetching neighbor vectors before computing their distances raised AVX2 throughput
  at ef 48 from about 10,300 to about 13,600 queries per second in single runs.

## Compression and re-ranking, Cohere-1M

`./build/release/bench/hnsw_bench --dataset cohere-1m --encoding sq8|pq --rerank ...`

1M vectors, 768 dimensions, cosine, M = 16, ef_construction = 200, k = 10, 1,000
queries, median of 3 runs. Quantizers are trained on 50,000 evenly spaced vectors.
Re-ranking reads full vectors from the memory-mapped dataset file, so they are not part
of the index memory.

| Index | Bytes per vector | Index memory | Build (s) |
|---|---|---|---|
| Full floats | 3,072 | 3,121 MiB | 438 |
| SQ8 | 768 (4x smaller) | 924 MiB | 269 |
| PQ, 96 pieces | 96 (32x smaller) | 283 MiB | 349 |
| PQ, 48 pieces | 48 (64x smaller) | 237 MiB | 261 |

Index memory includes the graph links, which are not compressed.

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

### Notes

- SQ8 keeps nearly all the accuracy and is faster than full floats, because each
  distance reads a quarter of the memory. A re-rank depth of 20 gives back the rest.
- PQ alone stops improving at about 0.57 recall no matter how wide the search, because
  its compression error is larger than the gaps between close neighbors. Re-ranking is
  what makes it usable: 200 re-scored candidates reach 0.955.
- 64x PQ (48 pieces) does not reach 0.95 on this dataset even with 400 re-scored
  candidates. 32x is the practical limit for plain PQ here.

## Durability, SIFT

`./build/release/bench/durability_bench insert` and `... recovery`

128 dimensions, M = 16, ef_construction = 200. The log and database live on the
laptop's NVMe SSD (ext4). Each insert is one vector; "database" means the whole
`Database::insert` (log, then HNSW insert), "log only" is the write-ahead log alone.

### Insert throughput by sync mode

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

- One fdatasync takes about 0.65 ms on this SSD, so syncing every record on its own
  tops out near 1,500 per second however many threads write.
- Group commit lets writers that arrive during a sync share the next one. With 16
  threads it reaches 9,702 durable inserts per second, 7.9x every-write, with the
  same guarantee: nothing is acknowledged before it is on disk.
- With no syncs, the HNSW insert itself is the limit (about 9,300 per second on one
  thread).

### Recovery time

A snapshot of 800,000 vectors (496 MiB, written in 0.62 s), plus a log of N
inserts written after it. Time to open the database:

| Log records since checkpoint | Snapshot load (s) | Replay, 1 thread (s) | Replay, 12 threads (s) |
|---|---|---|---|
| 0 | 0.38 | 0.00 | 0.00 |
| 20,000 | 0.40 | 5.39 | 2.36 |
| 50,000 | 0.40 | 14.05 | 5.83 |
| 100,000 | 0.42 | 27.58 | 11.92 |
| 200,000 | 0.49 | 60.59 | 24.61 |

- Loading the snapshot is fast (about 1.3 GB/s, file already in the page cache).
  Replaying the log is slow, because each record is a full HNSW insert.
- So recovery time is set by the checkpoint interval: about 0.12 ms per record
  since the last checkpoint with 12 threads. Since a checkpoint costs well under a
  second, checkpointing every 20,000 to 50,000 inserts keeps recovery under a few seconds.

### Crash testing

| Test | Result |
|---|---|
| Kill-9 crash test, group commit | 10,000 crashes, 188,462 acknowledged inserts checked, none lost or damaged |
| Kill-9 crash test, every-write | 3,000 crashes, 19,124 acknowledged inserts, none lost or damaged |
| Kill-9 crash test, no sync | 3,000 crashes, 139,621 acknowledged inserts, none lost or damaged |
| libFuzzer + AddressSanitizer, log reader and recovery | 1 hour, 18.6M inputs, no crashes |
| libFuzzer + AddressSanitizer, snapshot loader | 1 hour, 46.1M inputs, no crashes |
