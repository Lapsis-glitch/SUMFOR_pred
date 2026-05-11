"""Diagnose why fragment-per-spectrum distribution is skewed towards 1."""
import os, json, csv, sys
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from collections import Counter

BASE = "/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs"
RUN = "2026-04-14_17-02-56"  # latest run
run_dir = os.path.join(BASE, RUN)

# ── Load entry metrics ──
entry_counts = []
with open(os.path.join(run_dir, "entry_metrics.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        entry_counts.append(int(row["predicted_count"]))

arr = np.array(entry_counts)
print("=== Fragments per spectrum (above hybrid_thr) ===", flush=True)
print(f"  N entries: {len(arr)}", flush=True)
print(f"  Mean:   {arr.mean():.2f}", flush=True)
print(f"  Median: {np.median(arr):.1f}", flush=True)
print(f"  Std:    {arr.std():.2f}", flush=True)
print(f"  Min:    {arr.min()}", flush=True)
print(f"  Max:    {arr.max()}", flush=True)
for pct in [10, 25, 50, 75, 90, 95]:
    print(f"  P{pct:2d}: {np.percentile(arr, pct):.0f}", flush=True)

print("\n  Distribution:", flush=True)
for n in range(1, 21):
    count = np.sum(arr == n)
    if count > 0:
        print(f"    {n:2d} frags: {count:4d} entries ({100*count/len(arr):.1f}%)", flush=True)
count_20plus = np.sum(arr > 20)
if count_20plus:
    print(f"    20+ frags: {count_20plus:4d} entries ({100*count_20plus/len(arr):.1f}%)", flush=True)

# ── Now look at the RAW scores before thresholding ──
# Load fragment metrics to see the score distribution of ALL fragments
print("\n=== Score distribution of ALL predicted fragments ===", flush=True)
all_hybrid = []
all_ml = []
all_physics = []
with open(os.path.join(run_dir, "fragment_metrics.jsonl")) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("hybrid_score") is not None:
            all_hybrid.append(rec["hybrid_score"])
        if rec.get("ml_prob") is not None:
            all_ml.append(rec["ml_prob"])
        if rec.get("score") is not None:
            all_physics.append(rec["score"])

h = np.array(all_hybrid)
m = np.array(all_ml)
p = np.array(all_physics)
print(f"  Total fragments (above thr): {len(h)}", flush=True)
print(f"  Hybrid: mean={h.mean():.4f} med={np.median(h):.4f} min={h.min():.4f} max={h.max():.4f}", flush=True)
print(f"  ML:     mean={m.mean():.4f} med={np.median(m):.4f} min={m.min():.4f} max={m.max():.4f}", flush=True)
print(f"  Phys:   mean={p.mean():.4f} med={np.median(p):.4f} min={p.min():.4f} max={p.max():.4f}", flush=True)

# Hybrid score histogram
print("\n  Hybrid score histogram:", flush=True)
for lo in np.arange(0.70, 1.01, 0.05):
    hi = lo + 0.05
    n = np.sum((h >= lo) & (h < hi))
    print(f"    [{lo:.2f}, {hi:.2f}): {n:5d} ({100*n/len(h):.1f}%)", flush=True)

# ML prob histogram
print("\n  ML prob histogram:", flush=True)
for lo in np.arange(0.0, 1.01, 0.10):
    hi = lo + 0.10
    n = np.sum((m >= lo) & (m < hi))
    if n > 0:
        print(f"    [{lo:.2f}, {hi:.2f}): {n:5d} ({100*n/len(m):.1f}%)", flush=True)

print("\nDone.", flush=True)

