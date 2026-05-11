"""Diagnose WHY low-fragment entries have so few fragments.
Is it: (a) too few NIST peaks? (b) too few candidates? (c) ML/physics killing them?"""
import os, json, csv
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from Large_data import (
    MERGED, _build_mol, _extract_bde_data,
    FRAG_DEPTH, RULE_FLAGS, MIN_REL_INTENSITY, ALPHA, TOP_K,
    USE_MULTIPLICATIVE_HYBRID, USE_COMPLEMENTARY_LOSS_FILTER,
    USE_BEST_PER_PEAK, _filter_complementary_loss, _best_per_peak,
)
from formula import Formula
from hybrid_enumerator import HybridEnumerator
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from ml_correction_integration import apply_ml_correction
from utils import convert_assignments

# Load entry metrics to find low-fragment entries
BASE = "/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs"
RUN = "2026-04-14_17-02-56"
run_dir = os.path.join(BASE, RUN)

low_frag_entries = []
high_frag_entries = []
with open(os.path.join(run_dir, "entry_metrics.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        n = int(row["predicted_count"])
        eid = row["entry_id"]
        if n <= 2:
            low_frag_entries.append(eid)
        elif 8 <= n <= 12:
            high_frag_entries.append(eid)

print(f"Low-frag entries (≤2): {len(low_frag_entries)}", flush=True)
print(f"High-frag entries (8-12): {len(high_frag_entries)}", flush=True)

# Sample some from each group
import random
random.seed(42)
sample_low = random.sample(low_frag_entries, min(30, len(low_frag_entries)))
sample_high = random.sample(high_frag_entries, min(15, len(high_frag_entries)))


def diagnose_entry(eid):
    entry = MERGED[eid]
    nist = entry["nist_matches"][0] if entry["nist_matches"] else None
    if not nist:
        return None
    csv_entry = entry["csv_entry"]
    formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    if not formula_str:
        return None
    parent_formula = Formula.from_string(formula_str)

    nist_mz = nist["mz"]
    nist_int = nist["intensities"]
    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]

    # Count peaks at different thresholds
    peaks_05 = sum(1 for rI in rel_int if rI >= 0.05)
    peaks_10 = sum(1 for rI in rel_int if rI >= 0.10)
    peaks_20 = sum(1 for rI in rel_int if rI >= 0.20)

    # Filtered peaks at current threshold
    filtered = [(mz, I, rI) for mz, I, rI in zip(nist_mz, nist_int, rel_int) if rI >= MIN_REL_INTENSITY]
    if not filtered:
        return None
    mz_filt = [mz for mz, _, _ in filtered]
    int_filt = [I for _, I, _ in filtered]
    rel_filt = [rI for _, _, rI in filtered]

    # Build enumerator
    mol = _build_mol(entry)
    bde_data = _extract_bde_data(entry)
    has_bde = mol is not None and bde_data is not None

    if has_bde:
        peak_enum = HybridEnumerator(
            parent_formula, mol, bde_data,
            rule_flags=RULE_FLAGS, max_depth=FRAG_DEPTH,
            bde_threshold=120.0, bde_softness=50.0,
        )
    else:
        peak_enum = PeakDrivenEnumerator(
            parent_formula, rule_flags=RULE_FLAGS,
            max_depth=FRAG_DEPTH, auto_detect_rules=True,
        )

    total_lookup = sum(len(v) for v in peak_enum.lookup.values())

    # Count candidates per filtered peak
    candidates_per_peak = []
    for mz in mz_filt:
        nm = round(mz)
        cands = peak_enum.enumerate_for_peak(nm)
        candidates_per_peak.append(len(cands))

    # Run assignment
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)
    assignments = engine.assign_peaks(
        nist_peaks=list(zip(mz_filt, int_filt)),
        rel_intensities=rel_filt,
    )

    # ML correction
    apply_ml_correction(
        assignments,
        parent_name=csv_entry.get("name", ""),
        parent_formula_str=formula_str,
        nist_mz=nist_mz, nist_int=rel_int,
    )

    # Convert and score
    ad = convert_assignments(assignments)
    all_scores = []
    for a in ad:
        physics = a.get("score", 0.0)
        ml = a.get("ml_prob", 0.0)
        if USE_MULTIPLICATIVE_HYBRID:
            hybrid = (max(ml, 1e-9) ** 0.8) * (max(physics, 1e-9) ** 0.2)
        else:
            hybrid = ALPHA * physics + (1 - ALPHA) * ml
        a["hybrid_score"] = hybrid
        if a.get("best_formula"):
            all_scores.append({"hybrid": hybrid, "ml": ml, "physics": physics})

    # Count at various thresholds
    above_075 = sum(1 for s in all_scores if s["hybrid"] >= 0.75)
    above_070 = sum(1 for s in all_scores if s["hybrid"] >= 0.70)
    above_065 = sum(1 for s in all_scores if s["hybrid"] >= 0.65)
    above_050 = sum(1 for s in all_scores if s["hybrid"] >= 0.50)

    return {
        "eid": eid,
        "formula": formula_str,
        "peaks_total": len(nist_mz),
        "peaks_05": peaks_05,
        "peaks_10": peaks_10,
        "peaks_20": peaks_20,
        "lookup_total": total_lookup,
        "has_bde": has_bde,
        "candidates_per_peak": candidates_per_peak,
        "total_candidates": sum(candidates_per_peak),
        "total_assigned": len([s for s in all_scores]),
        "above_075": above_075,
        "above_070": above_070,
        "above_065": above_065,
        "above_050": above_050,
        "scores": all_scores,
    }


print("\n=== LOW-FRAGMENT ENTRIES (≤2 frags) — sample of 30 ===", flush=True)
print(f"{'EID':>6s} {'Formula':20s} {'BDE':>3s} {'Pk05':>4s} {'Pk10':>4s} {'Pk20':>4s} "
      f"{'Lookup':>6s} {'Cands':>5s} {'Assign':>6s} {'≥.75':>4s} {'≥.70':>4s} {'≥.65':>4s} {'≥.50':>4s}",
      flush=True)

low_stats = []
for eid in sample_low:
    r = diagnose_entry(eid)
    if r is None:
        continue
    low_stats.append(r)
    print(f"{r['eid']:>6s} {r['formula']:20s} {'Y' if r['has_bde'] else 'N':>3s} "
          f"{r['peaks_05']:4d} {r['peaks_10']:4d} {r['peaks_20']:4d} "
          f"{r['lookup_total']:6d} {r['total_candidates']:5d} {r['total_assigned']:6d} "
          f"{r['above_075']:4d} {r['above_070']:4d} {r['above_065']:4d} {r['above_050']:4d}",
          flush=True)

print("\n=== HIGH-FRAGMENT ENTRIES (8-12 frags) — sample of 15 ===", flush=True)
print(f"{'EID':>6s} {'Formula':20s} {'BDE':>3s} {'Pk05':>4s} {'Pk10':>4s} {'Pk20':>4s} "
      f"{'Lookup':>6s} {'Cands':>5s} {'Assign':>6s} {'≥.75':>4s} {'≥.70':>4s} {'≥.65':>4s} {'≥.50':>4s}",
      flush=True)

high_stats = []
for eid in sample_high:
    r = diagnose_entry(eid)
    if r is None:
        continue
    high_stats.append(r)
    print(f"{r['eid']:>6s} {r['formula']:20s} {'Y' if r['has_bde'] else 'N':>3s} "
          f"{r['peaks_05']:4d} {r['peaks_10']:4d} {r['peaks_20']:4d} "
          f"{r['lookup_total']:6d} {r['total_candidates']:5d} {r['total_assigned']:6d} "
          f"{r['above_075']:4d} {r['above_070']:4d} {r['above_065']:4d} {r['above_050']:4d}",
          flush=True)

# Summary
print("\n=== SUMMARY ===", flush=True)
for label, stats in [("Low-frag (≤2)", low_stats), ("High-frag (8-12)", high_stats)]:
    if not stats:
        continue
    peaks10 = np.array([s["peaks_10"] for s in stats])
    peaks05 = np.array([s["peaks_05"] for s in stats])
    lookups = np.array([s["lookup_total"] for s in stats])
    cands = np.array([s["total_candidates"] for s in stats])
    assigned = np.array([s["total_assigned"] for s in stats])
    a75 = np.array([s["above_075"] for s in stats])
    a70 = np.array([s["above_070"] for s in stats])
    bde_pct = 100 * sum(1 for s in stats if s["has_bde"]) / len(stats)

    print(f"\n  {label} (n={len(stats)}):", flush=True)
    print(f"    Peaks@10%: mean={peaks10.mean():.1f}  median={np.median(peaks10):.0f}", flush=True)
    print(f"    Peaks@5%:  mean={peaks05.mean():.1f}  median={np.median(peaks05):.0f}", flush=True)
    print(f"    Lookup:    mean={lookups.mean():.0f}  median={np.median(lookups):.0f}", flush=True)
    print(f"    Cands/ent: mean={cands.mean():.1f}  median={np.median(cands):.0f}", flush=True)
    print(f"    Assigned:  mean={assigned.mean():.1f}  median={np.median(assigned):.0f}", flush=True)
    print(f"    ≥0.75:     mean={a75.mean():.1f}  median={np.median(a75):.0f}", flush=True)
    print(f"    ≥0.70:     mean={a70.mean():.1f}  median={np.median(a70):.0f}", flush=True)
    print(f"    Has BDE:   {bde_pct:.0f}%", flush=True)

    # Score distribution for all fragments in this group
    all_ml = [s["ml"] for st in stats for s in st["scores"]]
    all_phys = [s["physics"] for st in stats for s in st["scores"]]
    if all_ml:
        ml_arr = np.array(all_ml)
        ph_arr = np.array(all_phys)
        print(f"    ML prob:   mean={ml_arr.mean():.3f}  median={np.median(ml_arr):.3f}", flush=True)
        print(f"    Physics:   mean={ph_arr.mean():.3f}  median={np.median(ph_arr):.3f}", flush=True)
        below_75 = np.sum(ml_arr < 0.75)
        print(f"    ML < 0.75: {below_75} ({100*below_75/len(ml_arr):.1f}%)", flush=True)

print("\nDone.", flush=True)

