# File format

Little-endian. Checksums are CRC32C. Code: [wal.cpp](../src/wal.cpp),
[snapshot.cpp](../src/snapshot.cpp), [database.cpp](../src/database.cpp).

## Database directory

| File | What it holds |
|---|---|
| `MANIFEST` | which snapshot is current, and the last log record it includes |
| `snapshot_<id>.snap` | the whole index as of some log record (`<id>` is 20 digits) |
| `wal_<lsn>.log` | log records, starting with record number `<lsn>` (20 digits) |
| `LOCK` | locked with `flock` while a program has the database open |
| `*.tmp` | a file being written; removed when the database is opened |

LSN: log sequence number, starting at 1.

## Log record

```
offset  size  field
0       4     payload length in bytes (at most 256 MiB)
4       4     CRC32C of: bytes 0-3, then bytes 8-16, then the payload
8       8     lsn
16      1     type (1 = insert)
17      n     payload
```

A log file is records back to back. Reading stops at the first record that is
cut off, fails its CRC, or whose lsn is not previous + 1.

### Insert payload

```
offset  size            field
0       8               row id of the first vector
8       4               number of vectors (count)
12      4               dim
16      count*dim*4     vectors, float32, as passed by the caller
```

Inserts over about 4 MiB are split into several records.

## Snapshot file

A 256-byte header, then five sections back to back.

### Header

```
offset  size  field
0       8     magic "STRSNAP1"
8       4     format version (1)
12      4     header size (256)
16      1     metric (0 = L2, 1 = inner product, 2 = cosine)
17      1     encoding (0 = float32, 1 = SQ8, 2 = PQ)
18      6     zero
24      8     dim
32      8     M
40      8     ef_construction
48      8     seed for node levels
56      8     capacity
64      8     count (number of vectors)
72      8     entry point: node id in the low 32 bits, its level in the high 32
80      8     lsn of the last log record included (0 if none)
88      8     code size: bytes per vector for SQ8 and PQ, 0 for float32
96      120   section table: 5 entries of {u64 offset, u64 size, u32 CRC32C, u32 zero}
216     36    zero
252     4     CRC32C of bytes 0-251
```

### Sections

| # | Section | Size in bytes | Contents |
|---|---|---|---|
| 0 | quantizer | float32: 0; SQ8: 2 * dim * 4; PQ: 256 * dim * 4 | SQ8: dim mins, then dim scales. PQ: for each piece, its 256 centroids |
| 1 | vectors | float32: count * dim * 4; else count * code size | one row per vector, no padding |
| 2 | levels | count | top layer of each node (0-30) |
| 3 | layer-0 links | count * (1 + 2M) * 4 | per node: link count, then 2M u32 slots |
| 4 | upper links | (sum of levels) * (1 + M) * 4 | per node with level > 0, layers 1 to level: link count, then M slots |

Unused link slots are zero. Cosine vectors are stored normalized.

Load checks: header CRC, section CRCs, section sizes, file size, and every link
(target exists and is on a high enough layer). Any failure rejects the file.

## MANIFEST

```
offset  size  field
0       8     magic "STRMANI1"
8       4     version (1)
12      4     zero
16      8     id of the current snapshot
24      8     lsn of the last log record in that snapshot
32      4     CRC32C of bytes 0-31
36      4     zero
```

## Writing a file safely

Snapshots and the MANIFEST:

1. write `name.tmp`, `fdatasync`;
2. `rename` to `name`;
3. `fsync` the directory.

New log files: create, `fsync` the directory, then write records.

## Checkpoint

1. Wait for in-flight inserts, record the last lsn L, start a new log file at L + 1.
2. Write the snapshot of all rows up to L.
3. Write the MANIFEST naming the new snapshot and L.
4. Delete the old snapshot and log files with records only up to L.

A crash at any step leaves a valid old or new MANIFEST with all files it needs.

## Opening (recovery)

1. Read the MANIFEST; delete `*.tmp` files and snapshots it doesn't name.
2. Load the snapshot; its lsn must match the MANIFEST.
3. Replay log records after that lsn, in order.
4. A cut-off or damaged record at the end of the newest log file is truncated.
   Damage anywhere else, or a missing log file, is an error.
