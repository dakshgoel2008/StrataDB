#!/usr/bin/env python3
"""Download benchmark datasets and convert them to .fvecs / .ivecs.

Creates (inside data/, which git ignores):
    data/<name>/base.fvecs         vectors to search through (not normalized)
    data/<name>/query.fvecs        query vectors
    data/<name>/groundtruth.ivecs  true 100 nearest ids for each query
    data/<name>/meta.json          distance type, sizes, source links

Usage:
    python bench/prepare_datasets.py sift
    python bench/prepare_datasets.py cohere-1m
    python bench/prepare_datasets.py --list
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from vecs import VecsWriter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOWNLOADS = DATA / "_downloads"

GT_K = 100       # neighbors kept per query
CHUNK = 50_000   # rows converted at a time, keeps memory use low

ANN = "http://ann-benchmarks.com"
ZILLIZ = "https://assets.zilliz.com/benchmark"

DATASETS = {
    "sift": {
        "kind": "hdf5", "metric": "l2",
        "urls": {"all": f"{ANN}/sift-128-euclidean.hdf5"},
    },
    "gist": {
        "kind": "hdf5", "metric": "l2",
        "urls": {"all": f"{ANN}/gist-960-euclidean.hdf5"},
    },
    "cohere-1m": {
        "kind": "parquet", "metric": "cosine",
        "urls": {
            "train": f"{ZILLIZ}/cohere_medium_1m/train.parquet",
            "test": f"{ZILLIZ}/cohere_medium_1m/test.parquet",
            "neighbors": f"{ZILLIZ}/cohere_medium_1m/neighbors.parquet",
        },
    },
}


def download(url: str) -> Path:
    """Download into data/_downloads, skipping files that are already complete."""
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    # Several datasets have a "train.parquet", so add the folder name in front.
    parent, name = url.rsplit("/", 2)[1:]
    dest = DOWNLOADS / f"{parent}__{name}"

    # ann-benchmarks.com blocks Python's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "StrataDB-bench/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and dest.stat().st_size == total:
            print(f"  cached   {dest.name}")
            return dest

        part = dest.with_suffix(dest.suffix + ".part")
        done = 0
        with open(part, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  download {dest.name}: {done / total:6.1%} of "
                          f"{total / 1e9:.2f} GB", end="", flush=True)
        print()
    if total and done != total:
        raise RuntimeError(f"incomplete download of {url}")
    # Rename only when finished, so a half download is never mistaken for a full one.
    part.rename(dest)
    return dest


def convert_hdf5(src: Path, out: Path) -> dict:
    import h5py

    with h5py.File(src, "r") as f:
        train, test, neighbors = f["train"], f["test"], f["neighbors"]
        with VecsWriter(out / "base.fvecs", np.float32) as w:
            for i in range(0, train.shape[0], CHUNK):
                w.write(train[i:i + CHUNK])
                print(f"\r  base     {w.rows:,} / {train.shape[0]:,}", end="", flush=True)
        print()
        with VecsWriter(out / "query.fvecs", np.float32) as w:
            w.write(test[:])
        with VecsWriter(out / "groundtruth.ivecs", np.int32) as w:
            w.write(neighbors[:, :GT_K])
        return {"n_base": int(train.shape[0]), "n_query": int(test.shape[0]),
                "dim": int(train.shape[1])}


def list_column_to_matrix(col) -> np.ndarray:
    """Turn a column of equal-length lists into a 2-D NumPy array."""
    import pyarrow as pa

    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    flat = col.flatten().to_numpy()
    rows = len(col)
    if rows == 0 or flat.size % rows != 0:
        raise ValueError("embedding lists have different lengths")
    return flat.reshape(rows, flat.size // rows)


def convert_parquet(files: dict, out: Path) -> dict:
    import pyarrow.parquet as pq

    # The 3 GB train file is one big block; these settings read it piece by
    # piece instead of loading it all into memory.
    pf = pq.ParquetFile(files["train"], buffer_size=64 << 20, pre_buffer=False)
    expected_id = 0
    with VecsWriter(out / "base.fvecs", np.float32) as w:
        for batch in pf.iter_batches(batch_size=CHUNK, columns=["id", "emb"]):
            ids = batch.column("id").to_numpy()
            # Neighbor ids point at row numbers, so ids must be 0, 1, 2, ...
            if not (ids == np.arange(expected_id, expected_id + len(ids))).all():
                raise ValueError("train ids are not in order; reordering is not supported")
            expected_id += len(ids)
            w.write(list_column_to_matrix(batch.column("emb")))
            print(f"\r  base     {w.rows:,} / {pf.metadata.num_rows:,}", end="", flush=True)
        print()
        n_base, dim = w.rows, w.dim

    test = pq.read_table(files["test"]).sort_by("id")
    nbrs = pq.read_table(files["neighbors"]).sort_by("id")
    if not np.array_equal(test.column("id").to_numpy(), nbrs.column("id").to_numpy()):
        raise ValueError("test and neighbors files have different query ids")

    with VecsWriter(out / "query.fvecs", np.float32) as w:
        w.write(list_column_to_matrix(test.column("emb")))
    gt = list_column_to_matrix(nbrs.column("neighbors_id"))[:, :GT_K]
    if gt.max() >= n_base:
        raise ValueError("ground truth points at ids outside the base set")
    with VecsWriter(out / "groundtruth.ivecs", np.int32) as w:
        w.write(gt.astype(np.int32))
    return {"n_base": n_base, "n_query": test.num_rows, "dim": dim}


def prepare(name: str) -> None:
    spec = DATASETS[name]
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}]")

    files = {key: download(url) for key, url in spec["urls"].items()}
    if spec["kind"] == "hdf5":
        shape = convert_hdf5(files["all"], out)
    else:
        shape = convert_parquet(files, out)

    meta = {"name": name, "metric": spec["metric"], "gt_k": GT_K,
            "source": spec["urls"], **shape}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote    {out.relative_to(ROOT)}/  "
          f"({shape['n_base']:,} x {shape['dim']}, {shape['n_query']:,} queries)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", metavar="DATASET")
    ap.add_argument("--list", action="store_true", help="list available datasets")
    args = ap.parse_args()

    if args.list or not args.datasets:
        for name, spec in DATASETS.items():
            print(f"{name:12} metric={spec['metric']}")
        return 0
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(unknown)} (see --list)")
    for name in args.datasets:
        prepare(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
