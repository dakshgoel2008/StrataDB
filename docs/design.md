# Design

Each entry: the choice, why, and the rejected alternative.

## Scope

- One in-memory HNSW index, durable through a write-ahead log and snapshots.
- Insert and search only. No deletes, updates, filters or segments.

## Distances

- Smaller is always closer: L2 is squared distance, inner product is `-dot`, cosine is `1 - dot`.
- Cosine vectors are normalized once at insert, so cosine search is inner product search.
  Rejected: dividing by both lengths in every distance call.

## HNSW

| Choice | Why | Rejected |
|---|---|---|
| Fixed capacity, allocated up front | Data never moves while other threads read it | Growable arrays (need a global lock or safe memory reclamation) |
| Layer-0 links in one flat array of `[count][2M slots]` blocks | Layer 0 is where searches spend their time; one multiply finds a block | `vector<vector<>>` (pointer chase per node) |
| Vectors padded to 64 bytes | Each vector starts on a cache line | Unpadded rows |
| Per-node mutex; searches copy links, then unlock | Locks are held for nanoseconds; inserts don't block searches | One reader-writer lock; lock-free link lists |
| Entry point and top level packed in one 64-bit atomic | Readers always see a matching pair | Two separate atomics |
| Node level from a hash of the node id | Same levels for any thread count | Shared random generator |
| Paper's neighbor-selection heuristic | Spreads links in different directions, keeps recall high | Closest-M selection |
| Exact duplicates only get leftover link slots | Duplicates pass the heuristic and can fill a list, cutting the graph | Plain heuristic |
| Visited marks with a 16-bit search number | No clearing per search; array cleared once per 65,535 searches | Bitset cleared every search |
| Prefetch all unvisited neighbors before computing distances | Memory loads overlap | Load on first use |

## SIMD kernels

- One source file per instruction set (scalar, auto-vectorized, AVX2, AVX-512), each
  with its own compiler flags. The CPU is checked at startup and the fastest
  supported kernels are used.
- Kernel files include only `<immintrin.h>` and C headers, so no shared inline
  function gets compiled with AVX2 and then called on a CPU without it.
- Rejected: `-march=native` for the whole build (binary fails on older CPUs).

## Compression

| Choice | Why | Rejected |
|---|---|---|
| SQ8: per-dimension min and scale, 1 byte per value | 4x smaller, almost no recall loss | Global range (wastes precision) |
| SQ8 query kept in float, rewritten once per search | Distance is one pass over the code bytes | Quantizing the query (loses accuracy, no speedup) |
| PQ: 256 centroids per piece, k-means++ start | Standard, 1 byte per piece | Random start (worse centroids) |
| PQ distance via an `m x 256` table per query | m lookups per distance | Decoding each code |
| New node's distances use its full vector; node-to-node distances decode both codes | Graph stays close to the full-precision graph | Codes on both sides |
| Re-rank top candidates with full vectors read from a memory-mapped file | Recovers recall; full vectors stay out of RAM | Keeping full vectors in the index |

## Durability

| Choice | Why | Rejected |
|---|---|---|
| Write-ahead log + periodic snapshots | Every insert is on disk before it is acknowledged; recovery = load + replay | Memory-mapped index (OS writes pages in any order; no consistent state after a crash) |
| Group commit (leader takes all queued records, one fdatasync) | 7.9x throughput at 16 threads, same guarantee | One fdatasync per insert |
| Three sync modes: every-write, group, none | Lets the user trade durability for speed; all three measured | Single mode |
| Row ids and log numbers assigned under one lock | Replay gives every row its original id | Ids assigned at index insert time |
| After a failed write or fdatasync, all writes stop until reopen | After a failed fsync the kernel may have dropped the pages; a retry can falsely succeed | Retrying the sync |
| Checkpoint pauses inserts only to drain in-flight ones; snapshot written while inserts continue; links to newer nodes left out | Searches never pause; short insert pause | Stop-the-world checkpoint |
| Snapshot and MANIFEST: write temp, fsync, rename, fsync directory | A crash leaves the complete old or complete new file | Writing in place |
| Snapshot ids never reused | MANIFEST can never point to a different file than intended | Reusing an id after a failed checkpoint |
| CRC32C on every record, header, section and the MANIFEST; hardware `crc32` when available | Detects damage; fast | No checksums; CRC32 (slower in software, no instruction) |
| Load validates every link | A damaged file cannot cause an out-of-bounds read | Trusting the file |
| Only the newest log file may end in a cut-off record; anything else is an error | Never silently drops data | Skipping damaged records |
| Log replayed with many threads | Faster recovery; ids and vectors identical, graph equivalent | Single-threaded replay |

## Testing

- Exact flat index as the reference; matches FAISS on SIFT-1M and Cohere-1M.
- All tests under ASan + UBSan and TSan, including concurrent insert/search/checkpoint.
- Kill-9 crash tester: child inserts and reports acknowledged rows; parent kills it at a random time, recovers, checks every row.
- In-memory disk for tests: injects failed syncs and short writes.
- libFuzzer + ASan on the log reader (with full recovery) and the snapshot loader.
  Fuzzing builds skip checksum checks so inputs reach the code after them.

## Limits

- No deletes or updates.
- Index must fit in memory.
- Capacity fixed at creation (can be raised on reopen).
- Recovery time grows with records since the last checkpoint (about 0.12 ms each, 12 threads).
- Little-endian only.
