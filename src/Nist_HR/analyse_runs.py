"""Analyse the last 4 validation runs: breakdown + false-positive patterns."""
import os, json, csv, sys
from collections import Counter, defaultdict

RUNS = [
    "2026-04-14_14-14-01",
    "2026-04-14_16-42-31",
    "2026-04-14_16-53-33",
    "2026-04-14_17-02-56",
]
BASE = "/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs"

print("=" * 80)
print("RUN COMPARISON")
print("=" * 80)

for run_id in RUNS:
    run_dir = os.path.join(BASE, run_id)
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        s = json.load(f)
    cfg = s["config"]
    c = s["counts"]
    m = s["metrics"]
    print(f"\n--- {run_id} ---")
    print(f"  hybrid_thr={cfg['hybrid_thr']}  best_per_peak={cfg['use_best_per_peak']}  "
          f"min_rel_int={cfg['min_rel_intensity']}  comp_loss={cfg['use_complementary_loss_filter']}  "
          f"mass_defect={cfg['use_mass_defect_filter']}")
    print(f"  Evaluated: {c['entries_evaluated']}/{c['entries_total']}  "
          f"Skipped: {c['entries_skipped']}")
    print(f"  Fragments predicted: {c['fragments_predicted']}")
    print(f"  Precision mean:   {m['precision_macro_mean']:.4f}")
    print(f"  Precision median: {m['precision_macro_median']:.4f}")
    duration = (
        __import__("datetime").datetime.fromisoformat(s["finished_at"]) -
        __import__("datetime").datetime.fromisoformat(s["started_at"])
    ).total_seconds()
    print(f"  Runtime: {duration:.0f}s ({duration/60:.1f}min)")

# ─── False positive analysis on the latest run ───────────────────
print("\n" + "=" * 80)
print("FALSE POSITIVE ANALYSIS (latest run: {})".format(RUNS[-1]))
print("=" * 80)

latest_dir = os.path.join(BASE, RUNS[-1])

# Load fragment-level metrics
fp_records = []
tp_records = []
with open(os.path.join(latest_dir, "fragment_metrics.jsonl")) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("is_correct"):
            tp_records.append(rec)
        else:
            fp_records.append(rec)

print(f"\nTotal fragments: {len(tp_records) + len(fp_records)}")
print(f"  True positives:  {len(tp_records)}")
print(f"  False positives: {len(fp_records)}")
print(f"  Overall precision: {len(tp_records)/(len(tp_records)+len(fp_records)):.4f}")

# ─── FP score distributions ──────────────────────────────────────
print("\n--- FP vs TP score distributions ---")
for label, recs in [("TP", tp_records), ("FP", fp_records)]:
    scores = [r["hybrid_score"] for r in recs if r["hybrid_score"] is not None]
    ml_probs = [r["ml_prob"] for r in recs if r["ml_prob"] is not None]
    physics = [r["score"] for r in recs if r["score"] is not None]
    if scores:
        import numpy as np
        s = np.array(scores)
        m = np.array(ml_probs) if ml_probs else np.array([0])
        p = np.array(physics) if physics else np.array([0])
        print(f"\n  {label} (n={len(scores)}):")
        print(f"    hybrid_score:  mean={s.mean():.4f}  median={np.median(s):.4f}  min={s.min():.4f}  max={s.max():.4f}")
        print(f"    ml_prob:       mean={m.mean():.4f}  median={np.median(m):.4f}  min={m.min():.4f}  max={m.max():.4f}")
        print(f"    physics_score: mean={p.mean():.4f}  median={np.median(p):.4f}  min={p.min():.4f}  max={p.max():.4f}")

# ─── FP: predicted m/z analysis ──────────────────────────────────
print("\n--- FP predicted m/z distribution ---")
fp_mz = [r["pred_mz"] for r in fp_records if r["pred_mz"] is not None]
if fp_mz:
    import numpy as np
    arr = np.array(fp_mz)
    print(f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
          f"min={arr.min():.1f}  max={arr.max():.1f}")
    # Bin by mass range
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    for lo, hi in bins:
        n = np.sum((arr >= lo) & (arr < hi))
        if n > 0:
            print(f"    m/z {lo:3d}-{hi:3d}: {n} FPs")

# ─── FP: mass defect analysis ────────────────────────────────────
print("\n--- Mass defect: FP vs TP ---")
for label, recs in [("TP", tp_records), ("FP", fp_records)]:
    defects = [abs(r["pred_mz"] - round(r["pred_mz"])) for r in recs if r["pred_mz"] is not None]
    if defects:
        import numpy as np
        d = np.array(defects)
        print(f"  {label}: mean_defect={d.mean():.4f}  median={np.median(d):.4f}  max={d.max():.4f}")

# ─── FP: formula patterns ────────────────────────────────────────
print("\n--- FP formula element patterns ---")
from collections import Counter
import re

def parse_elements(formula_str):
    """Extract element set from formula string."""
    if not formula_str:
        return set()
    return set(re.findall(r'[A-Z][a-z]?', formula_str))

fp_element_sets = Counter()
tp_element_sets = Counter()
for r in fp_records:
    els = frozenset(parse_elements(r.get("pred_formula", "")))
    fp_element_sets[els] += 1
for r in tp_records:
    els = frozenset(parse_elements(r.get("pred_formula", "")))
    tp_element_sets[els] += 1

print("\n  Top FP element sets:")
for els, count in fp_element_sets.most_common(10):
    tp_count = tp_element_sets.get(els, 0)
    ratio = count / (count + tp_count) if (count + tp_count) > 0 else 0
    print(f"    {','.join(sorted(els)):20s}  FP={count:4d}  TP={tp_count:4d}  FP_rate={ratio:.2%}")

# ─── FP: fragment rank distribution ──────────────────────────────
print("\n--- FP by fragment rank ---")
fp_ranks = Counter(r.get("fragment_rank", "?") for r in fp_records)
print("  rank: count")
for rank in sorted(fp_ranks.keys()):
    print(f"    {rank}: {fp_ranks[rank]}")

# ─── FP: which entries have the most FPs? ────────────────────────
print("\n--- Entries with most FPs (top 15) ---")
entry_fp = Counter(r["entry_id"] for r in fp_records)
entry_tp = Counter(r["entry_id"] for r in tp_records)

# Load entry metrics for formula info
entry_formulas = {}
with open(os.path.join(latest_dir, "entry_metrics.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        entry_formulas[row["entry_id"]] = row.get("parent_formula", "?")

for eid, fp_count in entry_fp.most_common(15):
    tp_count = entry_tp.get(eid, 0)
    formula = entry_formulas.get(eid, "?")
    total = fp_count + tp_count
    print(f"  Entry {eid:>5s}: {formula:20s}  FP={fp_count}  TP={tp_count}  "
          f"prec={tp_count/total:.2f}  total={total}")

# ─── Across-run comparison: entries that GAINED FPs ──────────────
print("\n--- Cross-run: entries that changed precision ---")
# Compare run 0 (baseline) vs run 3 (latest)
run0_dir = os.path.join(BASE, RUNS[0])
run3_dir = os.path.join(BASE, RUNS[-1])

def load_entry_prec(run_dir):
    precs = {}
    with open(os.path.join(run_dir, "entry_metrics.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            precs[row["entry_id"]] = {
                "precision": float(row["precision"]),
                "predicted": int(row["predicted_count"]),
                "correct": int(row["correct_count"]),
                "formula": row.get("parent_formula", "?"),
            }
    return precs

prec0 = load_entry_prec(run0_dir)
prec3 = load_entry_prec(run3_dir)

# Entries that lost precision
degraded = []
for eid in prec3:
    if eid in prec0:
        p0 = prec0[eid]["precision"]
        p3 = prec3[eid]["precision"]
        if p3 < p0 - 0.01:
            degraded.append((eid, p0, p3, prec3[eid]))

degraded.sort(key=lambda x: x[1] - x[2], reverse=True)
print(f"\n  Entries that LOST precision (run1→run4): {len(degraded)}")
for eid, p0, p3, info in degraded[:15]:
    print(f"    Entry {eid:>5s}: {info['formula']:20s}  "
          f"prec {p0:.2f}→{p3:.2f}  (pred={info['predicted']}, correct={info['correct']})")

# Entries that gained precision
improved = []
for eid in prec3:
    if eid in prec0:
        p0 = prec0[eid]["precision"]
        p3 = prec3[eid]["precision"]
        if p3 > p0 + 0.01:
            improved.append((eid, p0, p3, prec3[eid]))

print(f"\n  Entries that GAINED precision (run1→run4): {len(improved)}")

print("\nDone.")

