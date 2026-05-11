"""Quick per-stage profiler for slow entries."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from Large_data import (
    MERGED, _build_mol, _extract_bde_data,
    FRAG_DEPTH, RULE_FLAGS, MIN_REL_INTENSITY,
    _filter_complementary_loss, _best_per_peak,
    USE_BEST_PER_PEAK, USE_COMPLEMENTARY_LOSS_FILTER,
    USE_MULTIPLICATIVE_HYBRID, USE_MASS_DEFECT_FILTER,
    ALPHA, TOP_K,
)
from formula import Formula
from hybrid_enumerator import HybridEnumerator
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from ml_correction_integration import apply_ml_correction
from utils import convert_assignments

ENTRY_IDS = sys.argv[1:] if len(sys.argv) > 1 else ["222"]


def profile_entry(eid):
    print(f"\n{'='*60}", flush=True)
    entry = MERGED[eid]
    nist = entry["nist_matches"][0] if entry["nist_matches"] else None
    if not nist:
        print(f"Entry {eid}: no NIST match, skipping", flush=True)
        return
    csv_entry = entry["csv_entry"]
    formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    print(f"Entry {eid}: {formula_str}  ({csv_entry.get('name','?')})", flush=True)

    parent_formula = Formula.from_string(formula_str)
    mol = _build_mol(entry)
    bde_data = _extract_bde_data(entry)

    if mol:
        print(f"  atoms={mol.GetNumAtoms()}  bonds={mol.GetNumBonds()}", flush=True)
    if bde_data:
        weak = [b for b in bde_data if b["bde"] < 120.0]
        print(f"  BDE bonds={len(bde_data)}  weak(<120)={len(weak)}", flush=True)

    # --- Stage 1: Enumerator construction ---
    t0 = time.perf_counter()
    if mol is not None and bde_data:
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
    t1 = time.perf_counter()
    total_frags = sum(len(v) for v in peak_enum.lookup.values())
    print(f"  [1] Enumerator: {t1-t0:.3f}s  ({total_frags} fragments in lookup)", flush=True)

    # If it's a HybridEnumerator, show BDE tree size
    if hasattr(peak_enum, 'bde_enum'):
        bde_total = sum(len(v) for v in peak_enum.bde_enum.lookup.values())
        rule_total = sum(len(v) for v in peak_enum.rule_enum.lookup.values())
        print(f"       BDE frags={bde_total}  rule frags={rule_total}", flush=True)

    # --- Stage 2: Peak assignment ---
    nist_mz = nist["mz"]
    nist_int = nist["intensities"]
    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]
    filtered = [(mz, I, rI) for mz, I, rI in zip(nist_mz, nist_int, rel_int) if rI >= MIN_REL_INTENSITY]
    mz_filt = [mz for mz, _, _ in filtered]
    int_filt = [I for _, I, _ in filtered]
    rel_filt = [rI for _, _, rI in filtered]
    print(f"  peaks: {len(nist_mz)} total, {len(filtered)} filtered", flush=True)

    t2 = time.perf_counter()
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)
    assignments = engine.assign_peaks(
        nist_peaks=list(zip(mz_filt, int_filt)),
        rel_intensities=rel_filt,
    )
    t3 = time.perf_counter()
    print(f"  [2] Assignment: {t3-t2:.3f}s  ({len(assignments)} assignments)", flush=True)

    # --- Stage 3: ML correction ---
    t4 = time.perf_counter()
    apply_ml_correction(
        assignments,
        parent_name=csv_entry.get("name", ""),
        parent_formula_str=formula_str,
        nist_mz=nist_mz,
        nist_int=rel_int,
    )
    t5 = time.perf_counter()
    print(f"  [3] ML correction: {t5-t4:.3f}s", flush=True)

    # --- Stage 4: Hybrid scoring + filters ---
    t6 = time.perf_counter()
    ad = convert_assignments(assignments)
    for a in ad:
        physics = a.get("score", 0.0)
        ml = a.get("ml_prob", 0.0)
        if USE_MULTIPLICATIVE_HYBRID:
            a["hybrid_score"] = (max(ml, 1e-9) ** 0.8) * (max(physics, 1e-9) ** 0.2)
        else:
            a["hybrid_score"] = ALPHA * physics + (1 - ALPHA) * ml
    if USE_COMPLEMENTARY_LOSS_FILTER:
        ad = _filter_complementary_loss(ad, parent_formula)
    ad.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    ad = ad[:TOP_K]
    if USE_BEST_PER_PEAK:
        ad = _best_per_peak(ad)
    t7 = time.perf_counter()
    print(f"  [4] Scoring+filters: {t7-t6:.3f}s  ({len(ad)} final frags)", flush=True)

    print(f"  TOTAL: {t7-t0:.3f}s", flush=True)


for eid in ENTRY_IDS:
    profile_entry(eid)

print("\nDone.", flush=True)

