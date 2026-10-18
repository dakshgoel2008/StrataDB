#!/usr/bin/env python3
"""Measure hnswlib and FAISS on a prepared dataset: accuracy vs. speed.

For each ef_search value it records recall@k, queries per second and latency,
then saves results to bench/results/baselines/<dataset>-<lib>.json.

Usage:
    python bench/run_baselines.py sift
    python bench/run_baselines.py cohere-1m --libs hnswlib

Each query is a Python call, which adds a few microseconds. For a fully fair
comparison from C++, use bench/hnsw_bench.cpp.
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

from vecs import mmap_vecs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "bench" / "results" / "baselines"

BUILD_CHUNK = 100_000
EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
WARMUP_QUERIES = 200


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)  # copy, since file-backed rows are read-only
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class HnswlibIndex:
    name = "hnswlib"

    def __init__(self, metric, dim, n, M, ef_construction, threads):
        import hnswlib
        self.version = package_version("hnswlib")
        # With "cosine", hnswlib normalizes the vectors itself.
        space = {"l2": "l2", "cosine": "cosine"}[metric]
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=n, M=M, ef_construction=ef_construction,
                              random_seed=100)
        self.threads = threads

    def add(self, chunk, start_id):
        ids = np.arange(start_id, start_id + len(chunk))
        self.index.add_items(np.ascontiguousarray(chunk), ids, num_threads=self.threads)

    def prepare_search(self, ef, k):
        self.index.set_ef(max(ef, k))
        self.index.set_num_threads(1)

    def search_one(self, q, k):
        labels, _ = self.index.knn_query(q, k=k)
        return labels[0]


class FaissIndex:
    name = "faiss"

    def __init__(self, metric, dim, n, M, ef_construction, threads):
        import faiss
        self.faiss = faiss
        self.version = package_version("faiss-cpu")
        self.cosine = metric == "cosine"
        # Cosine similarity is the dot product of normalized vectors.
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.cosine else faiss.METRIC_L2
        self.index = faiss.IndexHNSWFlat(dim, M, faiss_metric)
        self.index.hnsw.efConstruction = ef_construction
        faiss.omp_set_num_threads(threads)

    def _prep(self, x):
        return normalized(x) if self.cosine else np.ascontiguousarray(x, dtype=np.float32)

    def add(self, chunk, start_id):
        self.index.add(self._prep(chunk))  # FAISS numbers vectors 0, 1, 2, ... in order

    def prepare_search(self, ef, k):
        self.index.hnsw.efSearch = max(ef, k)
        self.faiss.omp_set_num_threads(1)

    def search_one(self, q, k):
        _, labels = self.index.search(self._prep(q), k)
        return labels[0]


LIBS = {"hnswlib": HnswlibIndex, "faiss": FaissIndex}


def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = sum(len(set(f[:k]) & set(g[:k])) for f, g in zip(found, gt))
    return hits / (len(found) * k)


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(lib_cls, ds: str, args) -> dict:
    meta = json.loads((DATA / ds / "meta.json").read_text())
    base = mmap_vecs(DATA / ds / "base.fvecs")
    queries = mmap_vecs(DATA / ds / "query.fvecs")
    gt = mmap_vecs(DATA / ds / "groundtruth.ivecs", np.int32)

    if args.max_base:
        # Ground truth is for the full set, so recall is wrong here. Quick tests only.
        base = base[:args.max_base]
    if args.queries:
        queries, gt = queries[:args.queries], gt[:args.queries]
    n, dim = base.shape

    index = lib_cls(meta["metric"], dim, n, args.M, args.ef_construction, args.threads)
    print(f"[{ds} / {index.name} {index.version}] building on {n:,} x {dim} "
          f"(M={args.M}, ef_construction={args.ef_construction}, threads={args.threads})")
    t0 = time.perf_counter()
    for i in range(0, n, BUILD_CHUNK):
        index.add(base[i:i + BUILD_CHUNK], i)
        print(f"\r  added {min(i + BUILD_CHUNK, n):,} / {n:,}", end="", flush=True)
    build_s = time.perf_counter() - t0
    print(f"\n  build time {build_s:.1f} s")

    # Search on one fixed fast core, so results don't depend on which core the OS picks.
    allowed = os.sched_getaffinity(0)
    if args.search_cpu >= 0:
        os.sched_setaffinity(0, {args.search_cpu})

    points = []
    for ef in EF_SWEEP:
        if ef < args.k:
            continue
        index.prepare_search(ef, args.k)
        for q in queries[:WARMUP_QUERIES]:  # first queries are slow, so don't time them
            index.search_one(q[None, :], args.k)

        found = np.empty((len(queries), args.k), dtype=np.int64)
        lat_ns = np.empty(len(queries), dtype=np.int64)
        total0 = time.perf_counter_ns()
        for i, q in enumerate(queries):
            t = time.perf_counter_ns()
            found[i] = index.search_one(q[None, :], args.k)
            lat_ns[i] = time.perf_counter_ns() - t
        total_s = (time.perf_counter_ns() - total0) / 1e9

        p = np.percentile(lat_ns / 1e3, [50, 95, 99], method="inverted_cdf")
        point = {"ef_search": ef, "recall": recall_at_k(found, gt, args.k),
                 "qps": len(queries) / total_s,
                 "p50_us": float(p[0]), "p95_us": float(p[1]), "p99_us": float(p[2])}
        points.append(point)
        print(f"  ef={ef:4d}  recall@{args.k}={point['recall']:.4f}  "
              f"qps={point['qps']:8.0f}  p99={point['p99_us']:8.1f} us")
    os.sched_setaffinity(0, allowed)

    return {
        "dataset": ds, "library": index.name, "library_version": index.version,
        "metric": meta["metric"], "n_base": n, "dim": dim, "n_query": len(queries),
        "k": args.k, "M": args.M, "ef_construction": args.ef_construction,
        "build_threads": args.threads, "search_threads": 1, "search_cpu": args.search_cpu,
        "build_seconds": build_s,
        "subset": bool(args.max_base),
        "hardware": {"cpu": cpu_model(), "cores": os.cpu_count(),
                     "machine": platform.machine(), "python": platform.python_version()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "points": points,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--libs", nargs="+", default=list(LIBS), choices=list(LIBS))
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="threads used while building (search always uses 1)")
    ap.add_argument("--search-cpu", type=int, default=2,
                    help="CPU to run searches on (-1 = any); CPUs 0-3 are the fast cores")
    ap.add_argument("--queries", type=int, default=0, help="use only the first N queries")
    ap.add_argument("--max-base", type=int, default=0,
                    help="quick test on the first N vectors (recall will be wrong)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    for lib in args.libs:
        result = run(LIBS[lib], args.dataset, args)
        suffix = "-subset" if args.max_base else ""
        out = RESULTS / f"{args.dataset}-{lib}{suffix}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  saved {out.relative_to(ROOT)}")
        gc.collect()  # free this index before building the next one
    return 0


if __name__ == "__main__":
    sys.exit(main())
