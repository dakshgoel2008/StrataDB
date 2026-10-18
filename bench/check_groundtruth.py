#!/usr/bin/env python3
"""Check StrataDB's exact search against FAISS and the dataset's own answers.

Two answers can differ only when several vectors are exactly as far from the
query as the k-th result (a tie). Any other difference is a bug.

Usage:
    python bench/check_groundtruth.py sift
    python bench/check_groundtruth.py cohere-1m --k 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vecs import mmap_vecs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CHUNK = 200_000   # base rows given to FAISS at a time, keeps memory low
TIE_TOLERANCE = 1e-5  # relative; float rounding differs between libraries


def prepare(x: np.ndarray, metric: str) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x /= norms
    return x


def faiss_exact(base, queries, metric: str, k: int):
    """Exact top-k with FAISS, one chunk of the base at a time."""
    import faiss

    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    xq = prepare(queries, metric)
    best_d = np.full((len(xq), 0), np.inf, dtype=np.float32)
    best_i = np.zeros((len(xq), 0), dtype=np.int64)
    for start in range(0, len(base), CHUNK):
        d, i = faiss.knn(xq, prepare(base[start:start + CHUNK], metric), k, metric=faiss_metric)
        if metric != "l2":
            d = -d  # bigger dot product = closer, so flip it to "smaller = closer"
        d = np.hstack([best_d, d])
        i = np.hstack([best_i, i + start])
        keep = np.argsort(d, axis=1, kind="stable")[:, :k]
        best_d = np.take_along_axis(d, keep, axis=1)
        best_i = np.take_along_axis(i, keep, axis=1)
        print(f"\r  faiss    {min(start + CHUNK, len(base)):,} / {len(base):,}", end="", flush=True)
    print()
    return best_i


def exact_distances(base, query, ids, metric: str) -> np.ndarray:
    """Distances in float64 for a few rows, used to settle ties."""
    q = query.astype(np.float64)
    rows = base[np.asarray(ids)].astype(np.float64)
    if metric == "l2":
        return ((rows - q) ** 2).sum(axis=1)
    if metric == "cosine":
        q = q / (np.linalg.norm(q) or 1.0)
        norms = np.linalg.norm(rows, axis=1)
        norms[norms == 0] = 1.0
        return 1.0 - (rows / norms[:, None]) @ q
    return -(rows @ q)


def compare(name, ours, ref, base, queries, metric: str, k: int) -> int:
    """Prints how many queries match and returns the number that can't be explained by ties."""
    identical = same_order = ties = 0
    unexplained = []
    for qi in range(len(ours)):
        a, b = ours[qi, :k], ref[qi, :k]
        if np.array_equal(a, b):
            same_order += 1
        if set(a) == set(b):
            identical += 1
            continue
        # The answers may only differ in rows that are as far as our k-th result.
        boundary = exact_distances(base, queries[qi], [a[-1]], metric)[0]
        differing = list(set(a) ^ set(b))
        d = exact_distances(base, queries[qi], differing, metric)
        if np.all(np.abs(d - boundary) <= TIE_TOLERANCE * max(1.0, abs(boundary))):
            ties += 1
        else:
            unexplained.append(qi)

    n = len(ours)
    print(f"  vs {name:10} same set {identical / n:7.2%}   same order {same_order / n:7.2%}   "
          f"ties {ties}   unexplained {len(unexplained)}")
    if unexplained:
        print(f"    first unexplained queries: {unexplained[:10]}")
    return len(unexplained)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ours", default="groundtruth_strata", help="file prefix inside data/<dataset>/")
    ap.add_argument("--skip-faiss", action="store_true")
    args = ap.parse_args()

    folder = DATA / args.dataset
    metric = json.loads((folder / "meta.json").read_text())["metric"]
    base = mmap_vecs(folder / "base.fvecs")
    queries = mmap_vecs(folder / "query.fvecs")
    ours = mmap_vecs(folder / f"{args.ours}.ivecs", np.int32)
    given = mmap_vecs(folder / "groundtruth.ivecs", np.int32)
    k = min(args.k, ours.shape[1], given.shape[1])
    print(f"[{args.dataset}] metric {metric}, {len(queries):,} queries, k = {k}")

    failures = compare("dataset", ours, given, base, queries, metric, k)
    if not args.skip_faiss:
        failures += compare("faiss", ours, faiss_exact(base, queries, metric, k),
                            base, queries, metric, k)
    print("  OK: every difference is a tie" if failures == 0 else "  FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
