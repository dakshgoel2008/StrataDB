# File format

All numbers are little-endian. Every checksum is CRC32C (the Castagnoli
polynomial, the same one ext4 and RocksDB use). Code: [wal.cpp](../src/wal.cpp),
[snapshot.cpp](../src/snapshot.cpp), [database.cpp](../src/database.cpp).

## Database directory

| File | What it holds |
|---|---|
| `MANIFEST` | which snapshot is current, and the last log record it includes |
| `snapshot_<id>.snap` | the whole index as of some log record (`<id>` is 20 digits) |
| `wal_<lsn>.log` | log records, starting with record number `<lsn>` (20 digits) |
| `LOCK` | locked with `flock` while a program has the database open |
| `*.tmp` | a file being written; removed when the database is opened |

Log records are numbered 1, 2, 3, ... This number is the LSN (log sequence number).

## Log record

```
offset  size  field
0       4     payload length in bytes (at most 256 MiB)
4       4     CRC32C of: bytes 0-3, then bytes 8-16, then the payload
8       8     lsn
16      1     type (1 = insert)
17      n     payload
```

A log file is records back to back with nothing else. A reader stops at the
first record that is cut off, fails its CRC, or has the wrong lsn (each record
must be exactly one more than the one before).

### Insert payload

```
offset  size            field
0       8               row id of the first vector
8       4               number of vectors (count)
12      4               dim
16      count*dim*4     the vectors, as float32, exactly as the caller passed them
```

Inserts larger than about 4 MiB are split into several records.

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

Unused link slots are zero. For cosine, the stored vectors are already normalized.

Loading checks the header CRC, each section's CRC, that the section table matches
the sizes the header calls for, that the file has no extra bytes, and that every
link points to an existing node on a high enough layer. A damaged file is
refused, never half-loaded.

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

A snapshot or MANIFEST is never written in place:

1. write `name.tmp` and `fdatasync` it;
2. `rename` it to `name` (replacing the old file all at once);
3. `fsync` the directory, so the rename itself survives a system crash.

A new log file is created, then the directory is synced, before any record goes into it.

## Checkpoint

1. Wait for running inserts to finish, note the last lsn L, and start a new log
   file at L + 1. New inserts go on from here.
2. Write the snapshot of every row up to L, safely as above.
3. Write a MANIFEST that names the new snapshot and L, safely as above.
4. Delete the old snapshot and the log files that only hold records up to L.

A crash at any step leaves either the old MANIFEST (old snapshot plus the whole
log, all still there) or the new one (new snapshot plus the log after L).

## Opening (recovery)

1. Read the MANIFEST. Delete `*.tmp` files and snapshots it doesn't name.
2. Load the snapshot and check that its lsn matches the MANIFEST.
3. Replay the log in order, skipping records up to the snapshot's lsn.
4. If the newest log file ends in a cut-off or damaged record, cut the file there.
   That record was never acknowledged (or, with sync mode `none`, was lost in a
   system crash, which that mode allows). Damage anywhere else, or a missing log
   file, stops the open with an error instead of silently losing data.
