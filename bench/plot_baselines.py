#!/usr/bin/env python3
"""Plot recall vs. queries per second for all results of one dataset.

Usage:
    python bench/plot_baselines.py sift
    -> bench/results/plots/sift-recall-qps.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()

    files = sorted((RESULTS / "baselines").glob(f"{args.dataset}-*.json"))
    files = [f for f in files if not f.stem.endswith("-subset")]
    if not files:
        print(f"no results for '{args.dataset}' in {RESULTS / 'baselines'}")
        return 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in files:
        r = json.loads(f.read_text())
        pts = sorted(r["points"], key=lambda p: p["recall"])
        ax.plot([p["recall"] for p in pts], [p["qps"] for p in pts], marker="o",
                markersize=3, label=f"{r['library']} {r['library_version']}")

    ax.set_xlabel(f"recall@{r['k']}")
    ax.set_ylabel("queries per second (1 thread, log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{args.dataset}: {r['n_base']:,} x {r['dim']}, "
                 f"M={r['M']}, ef_construction={r['ef_construction']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out = RESULTS / "plots" / f"{args.dataset}-recall-qps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
