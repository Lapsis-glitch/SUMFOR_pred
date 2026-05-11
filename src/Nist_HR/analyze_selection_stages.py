"""
analyze_selection_stages.py

Quick post-hoc analysis for precision-oriented validation runs.

Reads `fragment_metrics.jsonl` from a run directory and reports fragment-level
precision split by selection stage (`strict` vs `rescued`), plus a few helpful
score summaries.

Usage:
    python analyze_selection_stages.py validation_outputs_precision/<run_id>/fragment_metrics.jsonl

If no argument is given, the script tries to use the most recently modified
`fragment_metrics.jsonl` under `validation_outputs_precision/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs_precision")


def _find_latest_fragment_metrics() -> Path | None:
    candidates = sorted(BASE_DIR.glob("*/fragment_metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _precision(rows):
    if not rows:
        return None
    correct = sum(1 for r in rows if r.get("is_correct"))
    return correct / len(rows)


def _describe_score(rows, key: str):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _print_stage(label: str, rows):
    print(f"\n--- {label} ---")
    print(f"count      : {len(rows)}")
    p = _precision(rows)
    print(f"precision  : {p:.4f}" if p is not None else "precision  : n/a")

    for key in ("ml_prob", "hybrid_score", "peak_posterior"):
        stats = _describe_score(rows, key)
        if stats:
            print(
                f"{key:11s}: mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}"
            )


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
    else:
        path = _find_latest_fragment_metrics()
        if path is None:
            raise SystemExit("No fragment_metrics.jsonl found under validation_outputs_precision/")

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = _load_jsonl(path)
    selected = [r for r in rows if r.get("selected_strict")]
    strict_core = [r for r in selected if r.get("selection_stage") == "strict"]
    rescued = [r for r in selected if r.get("selection_stage") == "rescued"]
    rejected = [r for r in rows if not r.get("selected_strict")]

    print("=" * 80)
    print(f"SELECTION STAGE ANALYSIS\nfile: {path}")
    print("=" * 80)
    print(f"all retained candidates : {len(rows)}")
    print(f"selected strict total   : {len(selected)}")
    print(f"strict core selected    : {len(strict_core)}")
    print(f"rescued selected        : {len(rescued)}")
    print(f"rejected retained       : {len(rejected)}")

    _print_stage("Selected total", selected)
    _print_stage("Strict core", strict_core)
    _print_stage("Rescued", rescued)

    if rescued:
        by_reason = {}
        for row in rescued:
            reason = row.get("rejection_reason") or "unknown"
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print("\nRescued fragment origin reasons:")
        for reason, count in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {reason}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()

