#!/usr/bin/env python3
"""Plot the results of durability_bench.

Usage:
    python bench/plot_durability.py sift
    -> bench/results/plots/sift-insert.png    inserts/s vs threads for each sync mode
    -> bench/results/plots/sift-recovery.png  recovery time vs log records since checkpoint
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def plot_inserts(dataset: str, out_dir: Path) -> bool:
    path = RESULTS / "durability" / f"insert-{dataset}.json"
    if not path.exists():
        print(f"missing {path}")
        return False
    data = json.loads(path.read_text())
    curves = defaultdict(list)
    for run in data["runs"]:
        curves[(run["target"], run["sync"])].append(
            (run["threads"], run["inserts"] / run["seconds"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, target in zip(axes, ["log only", "database"]):
        for (t, mode), pts in sorted(curves.items()):
            if t != target:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=mode)
        ax.set_xlabel("writer threads")
        ax.set_ylabel("inserts per second")
        ax.set_yscale("log")
        ax.set_title(f"{target} ({dataset}, {data['dim']} dims)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(title="sync mode")
    fig.tight_layout()
    out = out_dir / f"{dataset}-insert.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return True


def plot_recovery(dataset: str, out_dir: Path) -> bool:
    path = RESULTS / "durability" / f"recovery-{dataset}.json"
    if not path.exists():
        print(f"missing {path}")
        return False
    data = json.loads(path.read_text())
    curves = defaultdict(list)
    for p in data["points"]:
        curves[p["replay_threads"]].append(
            (p["log_records"], p["load_seconds"] + p["replay_seconds"], p["load_seconds"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for threads, pts in sorted(curves.items()):
        pts.sort()
        label = f"replay with {threads} thread" + ("s" if threads > 1 else "")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    load = min(p[2] for pts in curves.values() for p in pts)
    ax.axhline(load, color="gray", linestyle="--", label=f"snapshot load only ({load:.2f} s)")
    ax.set_xlabel("log records written since the last checkpoint")
    ax.set_ylabel("time to open after a crash (s)")
    ax.set_title(f"{dataset}: recovery time, snapshot of {data['snapshot_rows']:,} vectors")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"{dataset}-recovery.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()
    out_dir = RESULTS / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = plot_inserts(args.dataset, out_dir)
    ok = plot_recovery(args.dataset, out_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
