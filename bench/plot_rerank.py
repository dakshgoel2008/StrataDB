#!/usr/bin/env python3
"""Plot recall@k against re-rank depth for each compressed encoding.

For every re-rank depth the point with ef_search = max(depth, 64) is used.

Usage:
    python bench/plot_rerank.py cohere-1m
    -> bench/results/plots/cohere-1m-rerank.png
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
NAME = re.compile(r"strata-\w+?-(sq8|pq\d+)(?:-rerank(\d+))?$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()

    curves = defaultdict(list)
    k = 10
    for f in sorted((RESULTS / "baselines").glob(f"{args.dataset}-strata-*.json")):
        r = json.loads(f.read_text())
        match = NAME.search(r["library"])
        if not match:
            continue
        encoding, depth = match.group(1), int(match.group(2) or 0)
        target = max(depth, 64)
        point = min((p for p in r["points"] if p["ef_search"] >= target),
                    key=lambda p: p["ef_search"], default=None)
        if point:
            curves[encoding].append((depth, point["recall"]))
            k = r["k"]

    if not curves:
        print(f"no compressed results for '{args.dataset}'")
        return 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for encoding, pts in sorted(curves.items()):
        pts.sort()
        ax.plot([d for d, _ in pts], [rec for _, rec in pts], marker="o", label=encoding)
    ax.set_xlabel("re-rank depth (candidates re-scored with full vectors; 0 = none)")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"{args.dataset}: recall vs re-rank depth, ef_search = max(depth, 64)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = RESULTS / "plots" / f"{args.dataset}-rerank.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
