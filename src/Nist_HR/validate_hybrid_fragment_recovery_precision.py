"""
validate_hybrid_fragment_recovery_precision.py

Precision-oriented validation entry point for the SUMFOR hybrid fragment pipeline.

This file is intentionally separate from `validate_hybrid_fragment_recovery.py` so the
existing validator remains an untouched baseline.

What is different here:
  1. Rebuild the full candidate list locally (before baseline top-K / best-per-peak)
  2. Keep a small candidate shelf per nominal peak for analysis
  3. Create a strict export list with one-best-per-peak
  4. Apply explicit precision gates:
       - hybrid threshold
       - ML probability floor
       - optional top1-vs-top2 margin within the peak
  5. Export synthetic HR-like intensities from the original NIST peak intensities

Notes:
  - The richer `model_v2` feature expansion is intentionally ignored here. In practice
    it performed worse than the simpler setup, so this script focuses on selection /
    filtering rather than expanding the ML feature set.
  - This script is designed for precision-focused experiments while retaining candidate
    alternatives for later inspection.
"""

# ── Must come before any C-threaded library (numpy, lightgbm, …) ──
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
import multiprocessing as mp
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from Large_data import (
    MERGED,
    ENTRY_IDS,
    _build_mol,
    _extract_bde_data,
    RULE_FLAGS,
    FRAG_DEPTH,
    ALPHA,
    USE_MULTIPLICATIVE_HYBRID,
    USE_COMPLEMENTARY_LOSS_FILTER,
    USE_MASS_DEFECT_FILTER,
    _filter_complementary_loss,
    _filter_mass_defect,
)

MIN_REL_INTENSITY = 0.03
from chemical_classification import classify_molecule
from final_decision_calibrator import FinalDecisionCalibrator
from formula import Formula
from chemistry import exact_mass, dbe
from hybrid_enumerator import HybridEnumerator
from ml_correction_integration import apply_ml_correction
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from peak_driven_enumerator import PeakDrivenEnumerator
from utils import convert_assignments

# ── Matching / selection thresholds ──────────────────────────
TOL = 0.0003          # Da – fragment-to-AML matching tolerance
N_WORKERS = min(os.cpu_count() or 1, 16)

STRICT_SELECTION = {
    "hybrid_thr": 0.78,
    "ml_prob_floor": 0.65,
    "peak_softmax_temperature": 0.05,
    "peak_posterior_floor": 0.55,
    "min_margin_to_runner_up": None,
    "one_best_per_peak": True,
    "candidate_top_per_peak": 3,     # retained shelf for analysis / future rescue
    "candidate_hybrid_floor": None,  # None → retain top-N regardless of score
    "min_frags_per_entry": None,     # legacy unconditional rescue (replaced by calibrated top-up below)
    "topup_target_count": 7,         # calibrated top-up target frags per entry
    "topup_prob_floor": 0.92,        # decision_prob floor for top-up candidates
    "topup_max_expected_fdr": 0.05,  # cumulative expected FDR budget for top-up additions
    "rescue_hybrid_floor": 0.55,
    "rescue_ml_prob_floor": 0.50,
    "rescue_posterior_floor": 0.50,
    "rescue_max_expected_fdr": 0.30,
    "acceptance_score_weights": {
        "ml_prob": 0.45,
        "hybrid_score": 0.30,
        "peak_posterior": 0.15,
        "peak_rel_intensity": 0.10,
    },
    "ambiguity_penalties": {
        "per_extra_candidate": 0.0075,
        "max_candidate_penalty": 0.045,
        "small_gap_penalty": 0.02,
        "medium_gap_penalty": 0.01,
        "small_gap_thr": 0.03,
        "medium_gap_thr": 0.06,
    },
    "rule_family_penalties": {
        "bde": 0.010,
        "rearrangement": 0.010,
        "neutral": 0.005,
        "double": 0.006,
        "amine": 0.008,
        "amide": 0.006,
        "nitrile": 0.004,
    },
    "rule_family_mismatch_penalties": {
        "halogen": ("halogenated", 0.04),
        "sulfur": ("sulfur", 0.04),
        "phosphorus": ("phosphorus", 0.04),
        "amine": ("nitrogenous", 0.02),
        "nitrile": ("nitrogenous", 0.02),
        "amide": ("nitrogenous", 0.015),
    },
    "class_threshold_overrides": {
        "default": {
            "strict_acceptance_floor": 0.80,
            "rescue_acceptance_floor": 0.70,
            "rescue_max_expected_fdr": 0.30,
        },
        "halogenated": {
            "strict_acceptance_floor": 0.84,
            "rescue_acceptance_floor": 0.74,
            "rescue_max_expected_fdr": 0.24,
        },
        "sulfur": {
            "strict_acceptance_floor": 0.84,
            "rescue_acceptance_floor": 0.76,
            "rescue_max_expected_fdr": 0.22,
        },
        "phosphorus": {
            "strict_acceptance_floor": 0.85,
            "rescue_acceptance_floor": 0.77,
            "rescue_max_expected_fdr": 0.20,
        },
        "aromatic": {
            "strict_acceptance_floor": 0.79,
            "rescue_acceptance_floor": 0.70,
            "rescue_max_expected_fdr": 0.32,
        },
        "nitrogenous": {
            "strict_acceptance_floor": 0.81,
            "rescue_acceptance_floor": 0.72,
            "rescue_max_expected_fdr": 0.27,
        },
        "oxygenated": {
            "strict_acceptance_floor": 0.80,
            "rescue_acceptance_floor": 0.71,
            "rescue_max_expected_fdr": 0.28,
        },
    },
}

FINAL_DECISION_LAYER = {
    "enabled": False,
    "artifact_path": "final_decision_calibrator.pkl",
    "strict_prob_floor": 0.95,
    "rescue_prob_floor": 0.88,
    "use_prob_for_rescue_fdr": True,
}

RESULTS_OUTPUT = {
    "enabled": True,
    "detail": "entry_and_fragment",
    "formats": {
        "entry": "csv",
        "fragment": "jsonl",
        "run_summary": "json",
    },
    "out_dir": "validation_outputs_precision",
    "run_name": "auto",
    "include_fake_spectra": True,
    "fake_spectra_dirname": "fake_spectra_strict",
    "schema_version": "2.0",
    "mode": "precision_strict",
    "notes": [
        "Baseline validator remains unchanged in validate_hybrid_fragment_recovery.py.",
        "model_v2-style expanded feature sets were intentionally not revived here because they underperformed v1 in prior experiments.",
        "Strict export uses one-best-per-peak plus ML floor and peak-posterior gating.",
        "If too few fragments survive, a softer top1-only rescue stage refills the spectrum toward 5 fragments.",
        "Synthetic intensities come from the original NIST peak intensities.",
    ],
}


# ── Helper functions ─────────────────────────────────────────

def is_correct(frag_mz, aml_mz_list, tol=TOL):
    """Return True if *frag_mz* is within *tol* Da of any AML reference peak."""
    if frag_mz is None:
        return False
    return any(abs(frag_mz - m) <= tol for m in aml_mz_list)


def _safe_float(value):
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_stats(values):
    """Return (mean, max, min) for a list of nullable floats, or (None, None, None)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    arr = np.array(clean, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min())


def _hybrid_score(physics, ml):
    """Compute the current hybrid score used by this precision validator."""
    physics = 0.0 if physics is None else float(physics)
    ml = 0.0 if ml is None else float(ml)

    if USE_MULTIPLICATIVE_HYBRID:
        return (max(ml, 1e-9) ** 0.8) * (max(physics, 1e-9) ** 0.2)

    score = ALPHA * physics + (1 - ALPHA) * ml
    if physics < 0.1 and ml >= 0.8:
        score = ml
    return score


def _softmax(values, temperature):
    """Return a numerically stable softmax over *values* with temperature scaling."""
    if not values:
        return []

    temp = float(temperature) if temperature not in (None, 0) else 1.0
    temp = max(temp, 1e-6)
    arr = np.array(values, dtype=float) / temp
    arr -= arr.max()
    exp_arr = np.exp(arr)
    denom = exp_arr.sum()
    if denom <= 0:
        return [1.0 / len(values)] * len(values)
    return (exp_arr / denom).tolist()


# ── Per-fragment chemistry feature helpers ────────────────────
# These features are computed from the PREDICTED fragment formula + the
# PARENT formula + the NIST spectrum only. No AML reads — AML stays as the
# end-of-pipeline verification target.

_NEUTRAL_LOSS_WHITELIST = frozenset({
    "",            # no loss (parent molecular ion)
    "H1", "H2",
    "C1H3", "C1H4",
    "O1H1", "H2O1",
    "C1O1", "C1H1O1", "C1O2", "C1H1O2",
    "N1H2", "N1H3",
    "C1H1N1", "C1N1",
    "N1O1", "N1O2",
    "F1", "H1F1",
    "Cl1", "H1Cl1",
    "Br1", "H1Br1",
    "I1", "H1I1",
    "H2S1", "H1S1", "O1S1", "O2S1",
    "C2H2", "C2H3", "C2H4", "C2H5",
    "C3H5", "C3H6", "C3H7",
    "C4H8", "C4H9",
    "C6H5", "C6H6",
    "C1H2O1", "C1H3O1",
    "C2H2O1", "C2H4O1", "C2H3N1",
})

# Stable A+2 abundances (NIST / IUPAC monoisotopic ratios).
_AP2_ABUNDANCE = {"Cl": 0.3196, "Br": 0.9728, "S": 0.0443}
# Stable A+1 abundances we care about (mostly S; ¹³C is too noisy in NIST).
_AP1_ABUNDANCE = {"S": 0.0076}
_ISOTOPE_ELEMENTS = ("Cl", "Br", "S")


def _expected_isotope_ratio(frag_elems):
    """Theoretical A+2 over A intensity ratio from element counts."""
    ratio = 0.0
    for el in _ISOTOPE_ELEMENTS:
        n = frag_elems.get(el, 0)
        if n <= 0:
            continue
        ratio += n * _AP2_ABUNDANCE[el]
    return ratio


def _peak_intensity_near(nist_mz, nist_rel_int, target_mz, tol=0.6):
    """Return the relative intensity of the NIST peak nearest *target_mz* or 0.0."""
    best = 0.0
    for mz, rI in zip(nist_mz, nist_rel_int):
        if abs(mz - target_mz) <= tol:
            if rI > best:
                best = rI
    return best


def _isotope_consistency(frag_elems, nominal_mz, nist_mz, nist_rel_int):
    """
    Score how well the NIST [M+2] (and [M+1] for S) peak intensities match
    the theoretical isotope pattern for the predicted fragment formula.

    Returns 1.0 when:
      - the fragment has no isotope-bearing elements (neutral signal), or
      - the observed [M+2]/[M] ratio matches the theoretical ratio.

    Returns near 0.0 when expected isotope peaks are missing or grossly
    mismatched (strong FP signal for halogenated fragments).
    """
    has_isotopes = any(frag_elems.get(el, 0) > 0 for el in _ISOTOPE_ELEMENTS)
    if not has_isotopes:
        return 1.0
    if nominal_mz is None:
        return 1.0

    base_int = _peak_intensity_near(nist_mz, nist_rel_int, nominal_mz)
    if base_int <= 0.0:
        return 1.0

    theo = _expected_isotope_ratio(frag_elems)
    ap2_int = _peak_intensity_near(nist_mz, nist_rel_int, nominal_mz + 2)
    obs = ap2_int / base_int if base_int > 0 else 0.0

    n_S = frag_elems.get("S", 0)
    n_Cl = frag_elems.get("Cl", 0)
    n_Br = frag_elems.get("Br", 0)

    # Missing [M+2] is a strong FP signal when Cl/Br are claimed, but S
    # alone is too weak to flag (S [M+2] = 0.044 per S, often below noise).
    if obs <= 1e-4 and (n_Cl + n_Br) > 0:
        return 0.0
    if theo <= 1e-4:
        return 1.0

    # Log-ratio score, smooth and bounded in (0, 1].
    eps = 1e-3
    log_diff = abs(np.log((obs + eps) / (theo + eps)))
    score = float(np.exp(-log_diff))

    # If S is the only isotope contributor and [M+2] is missing, stay neutral.
    if n_Cl + n_Br == 0 and n_S > 0 and obs <= 1e-4:
        return 1.0
    return max(0.0, min(1.0, score))


def _neutral_loss_plausibility(parent_elems, frag_elems):
    """
    Classify the parent − fragment neutral loss against a curated EI whitelist.

    Returns:
      1.0  — loss is a well-known EI neutral (CH3, H2O, CO, HCN, etc.)
      0.5  — loss is a chemically plausible CHNO/CHN combination but not in the
             whitelist
      0.0  — loss is implausible (atom counts impossible, or only halogen lost
             from a non-halogen parent, etc.)
    """
    loss = {}
    for el, n in parent_elems.items():
        diff = n - frag_elems.get(el, 0)
        if diff < 0:
            return 0.0
        if diff > 0:
            loss[el] = diff
    # fragment contains element not in parent — implausible
    for el in frag_elems:
        if el not in parent_elems:
            return 0.0

    loss_key = Formula(loss).to_string() if loss else ""
    if loss_key in _NEUTRAL_LOSS_WHITELIST:
        return 1.0

    # graded path: small CHNO/CHN loss with non-negative DBE
    total_atoms = sum(loss.values())
    if total_atoms == 0:
        return 1.0  # already covered, defensive
    foreign = sum(1 for el in loss if el not in ("C", "H", "N", "O"))
    if foreign > 0:
        # halogen / S / P loss only acceptable if matching whitelist forms
        return 0.0
    try:
        loss_dbe = dbe(Formula(loss))
    except Exception:
        return 0.0
    if loss_dbe < -0.5:
        return 0.0
    if total_atoms <= 12:
        return 0.5
    return 0.0


def _mass_defect_norm(exact_mz):
    """Kendrick-style normalized mass defect: defect / nominal_mz."""
    if exact_mz is None:
        return 0.0
    nominal = round(exact_mz)
    if nominal <= 0:
        return 0.0
    return float((exact_mz - nominal) / nominal)


def _dbe_distance_to_parent(parent_dbe, frag_dbe):
    """Non-negative DBE drop from parent to fragment (capped at 0 below)."""
    if parent_dbe is None or frag_dbe is None:
        return 0.0
    return float(max(parent_dbe - frag_dbe, 0.0))


def _rescue_priority(candidate, meta_row):
    """Ranking key for controlled rescue of top1-per-peak rejected candidates."""
    return (
        meta_row.get("decision_prob", 0.0) or 0.0,
        meta_row.get("acceptance_score", 0.0) or 0.0,
        meta_row.get("peak_posterior", 0.0) or 0.0,
        candidate.get("ml_prob", 0.0) or 0.0,
        candidate.get("hybrid_score", 0.0) or 0.0,
        candidate.get("score", 0.0) or 0.0,
    )


def _resolve_parent_classes(parent_name, parent_formula):
    """Return sorted parent chemistry classes for threshold selection."""
    try:
        classes = classify_molecule(parent_name or "", parent_formula or "")
    except Exception:
        classes = {"unknown"}
    return sorted(classes) if classes else ["unknown"]


def _extract_rule_families(rule_source):
    """Extract a small, stable set of rule-family tags from a rule source string."""
    rs = (rule_source or "").lower()
    if not rs:
        return ["none"]

    families = []
    parts = [p.strip() for p in rs.split("+") if p.strip()]
    for part in parts:
        if "bde" in part:
            families.append("bde")
            continue
        token = part.split("(")[0].strip()
        token = token.split("_")[0].strip()
        if token:
            families.append(token)

    if not families:
        return ["none"]
    return sorted(set(families))


def _compute_rule_family_penalty(rule_families, parent_classes, config):
    """Penalty for historically noisier rule families and class-family mismatches."""
    penalties = config.get("rule_family_penalties", {})
    mismatch = config.get("rule_family_mismatch_penalties", {})
    total = 0.0
    details = []

    parent_classes = set(parent_classes or [])
    for family in rule_families:
        pen = penalties.get(family, 0.0)
        if pen:
            total += pen
            details.append(f"{family}:{pen:.3f}")
        if family in mismatch:
            required_class, extra_pen = mismatch[family]
            if required_class not in parent_classes:
                total += extra_pen
                details.append(f"{family}!{required_class}:{extra_pen:.3f}")

    return total, ";".join(details) if details else None


def _compute_ambiguity_penalty(n_candidates, gap_to_runner_up, config):
    """Penalty for crowded/ambiguous peaks."""
    params = config.get("ambiguity_penalties", {})
    total = 0.0
    details = []

    if n_candidates and n_candidates > 1:
        candidate_pen = min(
            (n_candidates - 1) * params.get("per_extra_candidate", 0.0),
            params.get("max_candidate_penalty", 0.0),
        )
        total += candidate_pen
        if candidate_pen:
            details.append(f"cand:{candidate_pen:.3f}")

    if gap_to_runner_up is not None:
        if gap_to_runner_up < params.get("small_gap_thr", 0.0):
            pen = params.get("small_gap_penalty", 0.0)
            total += pen
            if pen:
                details.append(f"gap_small:{pen:.3f}")
        elif gap_to_runner_up < params.get("medium_gap_thr", 0.0):
            pen = params.get("medium_gap_penalty", 0.0)
            total += pen
            if pen:
                details.append(f"gap_med:{pen:.3f}")

    return total, ";".join(details) if details else None


def _resolve_class_profile(parent_classes, config):
    """Resolve class-aware acceptance thresholds; stricter profiles win."""
    overrides = config.get("class_threshold_overrides", {})
    profile = dict(overrides.get("default", {}))
    profile_name = ["default"]
    for cls in parent_classes or []:
        override = overrides.get(cls)
        if not override:
            continue
        profile_name.append(cls)
        if "strict_acceptance_floor" in override:
            profile["strict_acceptance_floor"] = max(
                profile.get("strict_acceptance_floor", 0.0),
                override["strict_acceptance_floor"],
            )
        if "rescue_acceptance_floor" in override:
            profile["rescue_acceptance_floor"] = max(
                profile.get("rescue_acceptance_floor", 0.0),
                override["rescue_acceptance_floor"],
            )
        if "rescue_max_expected_fdr" in override:
            current = profile.get("rescue_max_expected_fdr")
            proposed = override["rescue_max_expected_fdr"]
            profile["rescue_max_expected_fdr"] = proposed if current is None else min(current, proposed)

    profile["profile_name"] = "+".join(profile_name)
    return profile


def _compute_acceptance_score(candidate, peak_posterior, gap_to_runner_up, n_candidates, rule_families, parent_classes, config):
    """Transparent final acceptance score from existing signals and penalties."""
    weights = config.get("acceptance_score_weights", {})
    ml_prob = candidate.get("ml_prob", 0.0) or 0.0
    hybrid = candidate.get("hybrid_score", 0.0) or 0.0
    rel_int = candidate.get("source_peak_rel_intensity", 0.0) or 0.0

    base_score = (
        weights.get("ml_prob", 0.0) * ml_prob +
        weights.get("hybrid_score", 0.0) * hybrid +
        weights.get("peak_posterior", 0.0) * (peak_posterior or 0.0) +
        weights.get("peak_rel_intensity", 0.0) * rel_int
    )

    rule_penalty, rule_penalty_detail = _compute_rule_family_penalty(rule_families, parent_classes, config)
    ambiguity_penalty, ambiguity_penalty_detail = _compute_ambiguity_penalty(n_candidates, gap_to_runner_up, config)
    acceptance_score = base_score - rule_penalty - ambiguity_penalty

    return {
        "base_score": _safe_float(base_score),
        "rule_family_penalty": _safe_float(rule_penalty),
        "ambiguity_penalty": _safe_float(ambiguity_penalty),
        "acceptance_score": _safe_float(acceptance_score),
        "rule_penalty_detail": rule_penalty_detail,
        "ambiguity_penalty_detail": ambiguity_penalty_detail,
    }


def _filter_rescue_pool_by_expected_fdr(candidates, meta, max_expected_fdr, use_decision_prob=False):
    """
    Keep the longest rescue prefix whose mean expected error (1 - ml_prob)
    stays below *max_expected_fdr*.

    Candidates are assumed to already be sorted from best to worst rescue priority.
    """
    if max_expected_fdr is None:
        for candidate in candidates:
            prob = meta[candidate["candidate_id"]].get("decision_prob") if use_decision_prob else None
            if prob is None:
                prob = candidate.get("ml_prob", 0.0) or 0.0
            meta[candidate["candidate_id"]]["rescue_fdr_expected"] = _safe_float(
                1.0 - prob
            )
            meta[candidate["candidate_id"]]["rescue_fdr_pass"] = True
        return candidates

    accepted = []
    cumulative_expected_fp = 0.0
    for candidate in candidates:
        prob = meta[candidate["candidate_id"]].get("decision_prob") if use_decision_prob else None
        if prob is None:
            prob = candidate.get("ml_prob", 0.0) or 0.0
        expected_fp = 1.0 - prob
        proposed_total = cumulative_expected_fp + expected_fp
        proposed_count = len(accepted) + 1
        proposed_fdr = proposed_total / proposed_count if proposed_count else 1.0

        meta[candidate["candidate_id"]]["rescue_fdr_expected"] = _safe_float(expected_fp)
        meta[candidate["candidate_id"]]["rescue_fdr_if_added"] = _safe_float(proposed_fdr)

        if proposed_fdr <= max_expected_fdr:
            accepted.append(candidate)
            cumulative_expected_fp = proposed_total
            meta[candidate["candidate_id"]]["rescue_fdr_pass"] = True
        else:
            meta[candidate["candidate_id"]]["rescue_fdr_pass"] = False

    return accepted


def _atomic_write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_jsonl(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _atomic_write_csv(path, rows):
    tmp = f"{path}.tmp"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_rows(path_base, fmt, rows):
    if not rows:
        return None
    if fmt == "csv":
        path = f"{path_base}.csv"
        _atomic_write_csv(path, rows)
        return path
    if fmt == "jsonl":
        path = f"{path_base}.jsonl"
        _atomic_write_jsonl(path, rows)
        return path
    raise ValueError(f"Unsupported format: {fmt}")


def save_validation_results(config, run_id, entry_records, fragment_records, run_summary):
    """Persist metrics and config snapshots to a run-specific output directory."""
    if not config.get("enabled", False):
        return {}

    run_dir = os.path.join(config.get("out_dir", "validation_outputs_precision"), run_id)
    os.makedirs(run_dir, exist_ok=True)

    written = {}

    entry_format = config.get("formats", {}).get("entry", "csv")
    entry_path = _write_rows(os.path.join(run_dir, "entry_metrics"), entry_format, entry_records)
    if entry_path:
        written["entry"] = entry_path

    if config.get("detail") == "entry_and_fragment":
        fragment_format = config.get("formats", {}).get("fragment", "jsonl")
        fragment_path = _write_rows(
            os.path.join(run_dir, "fragment_metrics"), fragment_format, fragment_records
        )
        if fragment_path:
            written["fragment"] = fragment_path

    config_path = os.path.join(run_dir, "config_snapshot.json")
    _atomic_write_json(config_path, config)
    written["config"] = config_path

    if config.get("formats", {}).get("run_summary", "json") == "json":
        summary_path = os.path.join(run_dir, "run_summary.json")
        _atomic_write_json(summary_path, run_summary)
        written["run_summary"] = summary_path

    return written


def _init_worker():
    """Silence RDKit's C++ logger inside worker processes."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


# ── Candidate generation without baseline top-K truncation ───

def run_single_entry_precision(entry_id: str):
    """
    Run the full pipeline for one entry but keep the full scored candidate list.

    Unlike `Large_data.run_single_entry`, this does not apply the baseline top-K or
    best-per-peak truncation because selection is handled locally in strict mode.
    """
    entry = MERGED[entry_id]

    csv_entry = entry["csv_entry"]
    aml_mz = csv_entry["mz"]
    aml_int = csv_entry["intensities"]

    if not entry["nist_matches"]:
        return None
    nist = entry["nist_matches"][0]
    nist_mz = nist["mz"]
    nist_int = nist["intensities"]
    if not nist_mz:
        return None

    parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    if not parent_formula_str:
        return None

    parent_name = csv_entry.get("name", "")
    parent_classes = _resolve_parent_classes(parent_name, parent_formula_str)

    parent_formula = Formula.from_string(parent_formula_str)

    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]
    nist_peaks_filtered = [
        (mz, I, rI)
        for mz, I, rI in zip(nist_mz, nist_int, rel_int)
        if rI >= MIN_REL_INTENSITY
    ]
    if not nist_peaks_filtered:
        return None

    mz_filt = [mz for mz, _, _ in nist_peaks_filtered]
    int_filt = [I for _, I, _ in nist_peaks_filtered]
    rel_filt = [rI for _, _, rI in nist_peaks_filtered]

    mol = _build_mol(entry)
    bde_data = _extract_bde_data(entry)
    if mol is not None and bde_data:
        peak_enum = HybridEnumerator(
            parent_formula,
            mol,
            bde_data,
            rule_flags=RULE_FLAGS,
            max_depth=FRAG_DEPTH,
            bde_threshold=120.0,
            bde_softness=50.0,
        )
    else:
        peak_enum = PeakDrivenEnumerator(
            parent_formula,
            rule_flags=RULE_FLAGS,
            max_depth=FRAG_DEPTH,
            auto_detect_rules=True,
        )

    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)
    assignments = engine.assign_peaks(
        nist_peaks=list(zip(mz_filt, int_filt)),
        rel_intensities=rel_filt,
    )

    apply_ml_correction(
        assignments,
        parent_name=csv_entry.get("name", ""),
        parent_formula_str=parent_formula_str,
        nist_mz=nist_mz,
        nist_int=rel_int,
    )

    assignments_dict = convert_assignments(assignments)
    parent_elems = parent_formula.elements
    parent_dbe = dbe(parent_formula)
    for idx, a in enumerate(assignments_dict):
        a["candidate_id"] = idx
        a["hybrid_score"] = _hybrid_score(a.get("score", 0.0), a.get("ml_prob", 0.0))
        peak_int = _safe_float(a.get("intensity"))
        a["source_peak_intensity"] = peak_int
        a["source_peak_rel_intensity"] = (
            peak_int / max_int if peak_int is not None and max_int > 0 else None
        )

        frag_str = a.get("best_formula")
        exact_mz = a.get("best_exact_mz")
        if frag_str and exact_mz is not None:
            try:
                frag_f = Formula.from_string(frag_str)
            except Exception:
                frag_f = None
            if frag_f is not None:
                frag_elems = frag_f.elements
                try:
                    a["frag_dbe"] = float(dbe(frag_f))
                except Exception:
                    a["frag_dbe"] = None
                a["mass_defect_abs"] = float(abs(exact_mz - round(exact_mz)))
                a["mass_defect_norm"] = _mass_defect_norm(exact_mz)
                a["dbe_distance_to_parent"] = _dbe_distance_to_parent(parent_dbe, a.get("frag_dbe"))
                a["neutral_loss_plausible"] = _neutral_loss_plausibility(parent_elems, frag_elems)
                a["isotope_consistency"] = _isotope_consistency(
                    frag_elems, a.get("nominal_mz"), nist_mz, rel_int
                )

    if USE_COMPLEMENTARY_LOSS_FILTER:
        assignments_dict = _filter_complementary_loss(assignments_dict, parent_formula)
    if USE_MASS_DEFECT_FILTER:
        assignments_dict = _filter_mass_defect(assignments_dict, parent_formula)

    assignments_dict.sort(
        key=lambda x: (
            x.get("hybrid_score", 0.0),
            x.get("ml_prob", 0.0),
            x.get("score", 0.0),
        ),
        reverse=True,
    )

    return {
        "entry_id": entry_id,
        "parent_name": parent_name,
        "parent_formula": parent_formula_str,
        "parent_classes": parent_classes,
        "nist_mz": nist_mz,
        "nist_int": nist_int,
        "aml_mz": aml_mz,
        "aml_int": aml_int,
        "assignments": assignments_dict,
    }


# ── Precision-oriented selection helpers ─────────────────────

def _select_precision_candidates(assignments, config, parent_classes=None, decision_layer=None):
    """
    Build two lists:
      - retained_candidates: top-N per peak for later analysis / rescue
      - strict_selected: one-best-per-peak after explicit precision gates

    Returns (retained_candidates, strict_selected, meta_by_candidate_id).
    """
    valid = [a for a in assignments if a.get("best_formula") is not None]
    grouped = defaultdict(list)
    for a in valid:
        grouped[a.get("nominal_mz")].append(a)

    retained = []
    strict = []
    meta = {}
    class_profile = _resolve_class_profile(parent_classes or ["unknown"], config)

    for nominal_mz, peak_candidates in grouped.items():
        peak_candidates = sorted(
            peak_candidates,
            key=lambda x: (
                x.get("hybrid_score", 0.0),
                x.get("ml_prob", 0.0),
                x.get("score", 0.0),
            ),
            reverse=True,
        )

        candidate_floor = config.get("candidate_hybrid_floor")
        if candidate_floor is not None:
            peak_candidates = [a for a in peak_candidates if a.get("hybrid_score", 0.0) >= candidate_floor]
            if not peak_candidates:
                continue

        top_n = config.get("candidate_top_per_peak")
        retained_here = peak_candidates if top_n is None else peak_candidates[:top_n]
        retained.extend(retained_here)

        posteriors = _softmax(
            [a.get("hybrid_score", 0.0) or 0.0 for a in peak_candidates],
            config.get("peak_softmax_temperature", 0.05),
        )
        posterior_by_candidate_id = {
            candidate["candidate_id"]: posterior
            for candidate, posterior in zip(peak_candidates, posteriors)
        }

        top = peak_candidates[0]
        runner_up = peak_candidates[1] if len(peak_candidates) > 1 else None
        gap = (
            top.get("hybrid_score", 0.0) - runner_up.get("hybrid_score", 0.0)
            if runner_up is not None else None
        )
        top_rule_families = _extract_rule_families(top.get("rule_source"))
        acceptance_parts = _compute_acceptance_score(
            top,
            posterior_by_candidate_id.get(top["candidate_id"], 0.0),
            gap,
            len(peak_candidates),
            top_rule_families,
            parent_classes or ["unknown"],
            config,
        )
        top_meta_seed = {
            "peak_posterior": _safe_float(posterior_by_candidate_id.get(top["candidate_id"])),
            "rule_families": top_rule_families,
            "rule_family_penalty": acceptance_parts["rule_family_penalty"],
            "ambiguity_penalty": acceptance_parts["ambiguity_penalty"],
            "acceptance_base_score": acceptance_parts["base_score"],
            "acceptance_score": acceptance_parts["acceptance_score"],
            "n_candidates_for_peak": len(peak_candidates),
            "gap_to_runner_up": _safe_float(gap),
        }
        decision_prob = None
        if decision_layer and decision_layer.get("calibrator") is not None:
            decision_prob = decision_layer["calibrator"].predict_proba(
                top,
                top_meta_seed,
                parent_classes or ["unknown"],
            )

        for rank, candidate in enumerate(retained_here, start=1):
            reason = None
            selected = False
            if rank != 1 and config.get("one_best_per_peak", True):
                reason = "not_top1_for_peak"
            elif rank == 1:
                ml_prob = candidate.get("ml_prob", 0.0) or 0.0
                hybrid = candidate.get("hybrid_score", 0.0) or 0.0
                peak_posterior = posterior_by_candidate_id.get(candidate["candidate_id"], 0.0)
                posterior_floor = config.get("peak_posterior_floor")
                min_gap = config.get("min_margin_to_runner_up")
                acceptance_score = acceptance_parts["acceptance_score"] or 0.0
                strict_acceptance_floor = class_profile.get("strict_acceptance_floor", 0.0)
                strict_prob_floor = (decision_layer or {}).get("strict_prob_floor", 0.0)

                if hybrid < config.get("hybrid_thr", 0.0):
                    reason = "below_hybrid_thr"
                elif ml_prob < config.get("ml_prob_floor", 0.0):
                    reason = "below_ml_floor"
                elif posterior_floor is not None and peak_posterior < posterior_floor:
                    reason = "below_peak_posterior"
                elif runner_up is not None and min_gap is not None and gap is not None and gap < min_gap:
                    reason = "margin_too_small"
                elif decision_prob is not None and decision_prob < strict_prob_floor:
                    reason = "below_decision_prob_floor"
                elif acceptance_score < strict_acceptance_floor:
                    reason = "below_acceptance_floor"
                else:
                    selected = True

            meta[candidate["candidate_id"]] = {
                "nominal_mz": nominal_mz,
                "rank_within_peak": rank,
                "n_candidates_for_peak": len(peak_candidates),
                "gap_to_runner_up": _safe_float(gap) if rank == 1 else None,
                "peak_posterior": _safe_float(posterior_by_candidate_id.get(candidate["candidate_id"])),
                "rule_families": top_rule_families if rank == 1 else _extract_rule_families(candidate.get("rule_source")),
                "rule_family_penalty": acceptance_parts["rule_family_penalty"] if rank == 1 else None,
                "ambiguity_penalty": acceptance_parts["ambiguity_penalty"] if rank == 1 else None,
                "rule_penalty_detail": acceptance_parts["rule_penalty_detail"] if rank == 1 else None,
                "ambiguity_penalty_detail": acceptance_parts["ambiguity_penalty_detail"] if rank == 1 else None,
                "acceptance_base_score": acceptance_parts["base_score"] if rank == 1 else None,
                "acceptance_score": acceptance_parts["acceptance_score"] if rank == 1 else None,
                "decision_prob": _safe_float(decision_prob) if rank == 1 else None,
                "threshold_profile": class_profile.get("profile_name"),
                "strict_selected": selected,
                "rejection_reason": reason,
                "retained_candidate": True,
                "selection_stage": "strict" if selected else None,
                "rescued_rank": None,
                "rescue_fdr_expected": None,
                "rescue_fdr_if_added": None,
                "rescue_fdr_pass": None,
            }

            if selected:
                strict.append(candidate)

    min_frags = config.get("min_frags_per_entry")
    if min_frags is not None and len(strict) < min_frags:
        already = {a["candidate_id"] for a in strict}
        rescue_pool_all = [
            a for a in retained
            if a["candidate_id"] not in already
            and meta[a["candidate_id"]]["rank_within_peak"] == 1
            and (a.get("hybrid_score", 0.0) or 0.0) >= config.get("rescue_hybrid_floor", 0.0)
            and (a.get("ml_prob", 0.0) or 0.0) >= config.get("rescue_ml_prob_floor", 0.0)
            and (meta[a["candidate_id"]].get("peak_posterior", 0.0) or 0.0) >= config.get("rescue_posterior_floor", 0.0)
            and (meta[a["candidate_id"]].get("acceptance_score", 0.0) or 0.0) >= class_profile.get("rescue_acceptance_floor", 0.0)
            and (
                meta[a["candidate_id"]].get("decision_prob") is None
                or (meta[a["candidate_id"]].get("decision_prob", 0.0) or 0.0) >= (decision_layer or {}).get("rescue_prob_floor", 0.0)
            )
        ]
        rescue_pool_all.sort(
            key=lambda x: _rescue_priority(x, meta[x["candidate_id"]]),
            reverse=True,
        )

        rescue_pool = _filter_rescue_pool_by_expected_fdr(
            rescue_pool_all,
            meta,
            class_profile.get("rescue_max_expected_fdr", config.get("rescue_max_expected_fdr")),
            use_decision_prob=bool((decision_layer or {}).get("use_prob_for_rescue_fdr", False)),
        )
        needed = max(0, min_frags - len(strict))
        for rescue_rank, candidate in enumerate(rescue_pool[:needed], start=1):
            strict.append(candidate)
            meta[candidate["candidate_id"]]["strict_selected"] = True
            meta[candidate["candidate_id"]]["rejection_reason"] = "rescued_for_min_frags"
            meta[candidate["candidate_id"]]["selection_stage"] = "rescued"
            meta[candidate["candidate_id"]]["rescued_rank"] = rescue_rank

    strict.sort(
        key=lambda x: (
            x.get("hybrid_score", 0.0),
            x.get("ml_prob", 0.0),
            x.get("score", 0.0),
        ),
        reverse=True,
    )
    retained.sort(
        key=lambda x: (
            x.get("nominal_mz", -1),
            meta[x["candidate_id"]]["rank_within_peak"],
            -(x.get("hybrid_score", 0.0)),
        )
    )
    return retained, strict, meta


def export_fake_spectrum(entry_id, parent_name, parent_formula, selected_assignments, out_dir):
    """Export the strict selected fragments as a synthetic HR-like spectrum JSON."""
    data = {
        "entry_id": entry_id,
        "parent_name": parent_name,
        "parent_formula": parent_formula,
        "mode": "strict_one_best_per_peak",
        "predicted_fragments": [
            {
                "mz": a.get("best_exact_mz"),
                "formula": a.get("best_formula"),
                "hybrid_score": a.get("hybrid_score"),
                "ml_prob": a.get("ml_prob"),
                "physics_score": a.get("score"),
                "peak_posterior": a.get("peak_posterior"),
                "selection_stage": a.get("selection_stage"),
                "rescued_rank": a.get("rescued_rank"),
                "nominal_mz": a.get("nominal_mz"),
                "intensity": a.get("source_peak_intensity"),
                "relative_intensity": a.get("source_peak_rel_intensity"),
                "rule_source": a.get("rule_source"),
            }
            for a in selected_assignments
        ],
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"entry_{entry_id}.json")
    _atomic_write_json(path, data)
    return path


# ── Main validation loop ─────────────────────────────────────

def main():
    precisions = []
    predicted_counts = []
    correct_counts = []
    entry_records = []
    fragment_records = []
    fake_spectrum_paths = []

    run_start = time.time()
    run_id = RESULTS_OUTPUT.get("run_name", "auto")
    if run_id == "auto":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Running precision-oriented hybrid validation...")
    print(f"  Multiplicative hybrid: {USE_MULTIPLICATIVE_HYBRID}")
    print(f"  Complementary-loss filter: {USE_COMPLEMENTARY_LOSS_FILTER}")
    print(f"  Mass-defect filter: {USE_MASS_DEFECT_FILTER}")
    print(f"  Strict hybrid threshold: {STRICT_SELECTION['hybrid_thr']}")
    print(f"  Strict ML floor: {STRICT_SELECTION['ml_prob_floor']}")
    print(f"  Peak posterior floor: {STRICT_SELECTION['peak_posterior_floor']}")
    print(f"  Peak softmax temperature: {STRICT_SELECTION['peak_softmax_temperature']}")
    print(f"  Strict min margin: {STRICT_SELECTION['min_margin_to_runner_up']}")
    print(f"  Strict min fragments (legacy rescue): {STRICT_SELECTION['min_frags_per_entry']}")
    print(f"  Calibrated top-up target count: {STRICT_SELECTION.get('topup_target_count')}")
    print(f"  Calibrated top-up prob floor: {STRICT_SELECTION.get('topup_prob_floor')}")
    print(f"  Calibrated top-up max expected FDR: {STRICT_SELECTION.get('topup_max_expected_fdr')}")
    print(f"  Min relative intensity (shadowed): {MIN_REL_INTENSITY}")
    print(f"  Default strict acceptance floor: {STRICT_SELECTION['class_threshold_overrides']['default']['strict_acceptance_floor']}")
    print(f"  Default rescue acceptance floor: {STRICT_SELECTION['class_threshold_overrides']['default']['rescue_acceptance_floor']}")
    print(f"  Final decision layer enabled: {FINAL_DECISION_LAYER['enabled']}")
    print(f"  Candidate shelf top-N per peak: {STRICT_SELECTION['candidate_top_per_peak']}")
    print(f"  Workers: {N_WORKERS}")

    decision_layer = dict(FINAL_DECISION_LAYER)
    decision_layer["calibrator"] = None
    if FINAL_DECISION_LAYER.get("enabled"):
        try:
            decision_layer["calibrator"] = FinalDecisionCalibrator(FINAL_DECISION_LAYER.get("artifact_path", "final_decision_calibrator.pkl"))
            print(f"  Final decision artifact: {FINAL_DECISION_LAYER.get('artifact_path', 'final_decision_calibrator.pkl')}")
        except Exception as exc:
            print(f"  Final decision artifact unavailable; falling back to heuristic mode ({exc})")
            decision_layer["enabled"] = False

    with mp.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(run_single_entry_precision, ENTRY_IDS)

        for result in tqdm(results_iter, total=len(ENTRY_IDS), desc="Validating", unit="entry"):
            if result is None:
                continue

            eid = result["entry_id"]
            parent_formula = result["parent_formula"]
            parent_classes = result.get("parent_classes", ["unknown"])
            aml_mz = result["aml_mz"]
            assignments = result["assignments"]

            retained_candidates, strict_selected, meta = _select_precision_candidates(
                assignments,
                STRICT_SELECTION,
                parent_classes=parent_classes,
                decision_layer=decision_layer,
            )

            strict_correct = 0
            candidate_true = 0
            strict_selected_ids = {a["candidate_id"] for a in strict_selected}

            for rank, a in enumerate(retained_candidates, start=1):
                candidate_id = a["candidate_id"]
                info = meta.get(candidate_id, {})
                frag_mz_val = _safe_float(a.get("best_exact_mz"))
                matched = is_correct(frag_mz_val, aml_mz)
                if matched:
                    candidate_true += 1
                if candidate_id in strict_selected_ids and matched:
                    strict_correct += 1

                if RESULTS_OUTPUT.get("enabled") and RESULTS_OUTPUT.get("detail") == "entry_and_fragment":
                    nominal = a.get("nominal_mz")

                    closest_aml = None
                    mass_error_da = None
                    mass_error_ppm = None
                    if frag_mz_val is not None and aml_mz:
                        closest_aml = min(aml_mz, key=lambda m: abs(m - frag_mz_val))
                        mass_error_da = frag_mz_val - closest_aml
                        if closest_aml > 0:
                            mass_error_ppm = (mass_error_da / closest_aml) * 1e6

                    neutral_loss_da = None
                    if frag_mz_val is not None:
                        try:
                            parent_mass = exact_mass(Formula.from_string(parent_formula))
                            neutral_loss_da = parent_mass - frag_mz_val
                        except Exception:
                            pass

                    mass_defect = None
                    if frag_mz_val is not None:
                        mass_defect = frag_mz_val - round(frag_mz_val)

                    frag_dbe = None
                    try:
                        frag_f = Formula.from_string(a.get("best_formula", ""))
                        frag_dbe = dbe(frag_f)
                    except Exception:
                        pass

                    selected_strict = bool(info.get("strict_selected", False))
                    selection_stage = info.get("selection_stage")
                    export_intensity = a.get("source_peak_intensity") if selected_strict else None
                    export_rel_intensity = a.get("source_peak_rel_intensity") if selected_strict else None
                    a["peak_posterior"] = info.get("peak_posterior")
                    a["selection_stage"] = selection_stage
                    a["rescued_rank"] = info.get("rescued_rank")

                    fragment_records.append({
                        "schema_version": RESULTS_OUTPUT.get("schema_version", "2.0"),
                        "run_id": run_id,
                        "entry_id": eid,
                        "parent_formula": parent_formula,
                        "parent_classes": parent_classes,
                        "candidate_global_rank": rank,
                        "nominal_mz": nominal,
                        "rank_within_peak": info.get("rank_within_peak"),
                        "n_candidates_for_peak": info.get("n_candidates_for_peak"),
                        "gap_to_runner_up": _safe_float(info.get("gap_to_runner_up")),
                        "peak_posterior": _safe_float(info.get("peak_posterior")),
                        "rule_families": info.get("rule_families"),
                        "rule_family_penalty": _safe_float(info.get("rule_family_penalty")),
                        "ambiguity_penalty": _safe_float(info.get("ambiguity_penalty")),
                        "acceptance_base_score": _safe_float(info.get("acceptance_base_score")),
                        "acceptance_score": _safe_float(info.get("acceptance_score")),
                        "decision_prob": _safe_float(info.get("decision_prob")),
                        "threshold_profile": info.get("threshold_profile"),
                        "rule_penalty_detail": info.get("rule_penalty_detail"),
                        "ambiguity_penalty_detail": info.get("ambiguity_penalty_detail"),
                        "retained_candidate": info.get("retained_candidate", False),
                        "selected_strict": selected_strict,
                        "selection_stage": selection_stage,
                        "rescued_rank": info.get("rescued_rank"),
                        "rescue_fdr_expected": _safe_float(info.get("rescue_fdr_expected")),
                        "rescue_fdr_if_added": _safe_float(info.get("rescue_fdr_if_added")),
                        "rescue_fdr_pass": info.get("rescue_fdr_pass"),
                        "rejection_reason": info.get("rejection_reason"),
                        "pred_formula": a.get("best_formula"),
                        "pred_mz": frag_mz_val,
                        "mass_defect": _safe_float(mass_defect),
                        "mass_defect_abs": _safe_float(a.get("mass_defect_abs")),
                        "mass_defect_norm": _safe_float(a.get("mass_defect_norm")),
                        "neutral_loss_plausible": _safe_float(a.get("neutral_loss_plausible")),
                        "isotope_consistency": _safe_float(a.get("isotope_consistency")),
                        "dbe_distance_to_parent": _safe_float(a.get("dbe_distance_to_parent")),
                        "frag_dbe": _safe_float(frag_dbe),
                        "neutral_loss_da": _safe_float(neutral_loss_da),
                        "rule_source": a.get("rule_source"),
                        "physics_score": _safe_float(a.get("score")),
                        "ml_prob": _safe_float(a.get("ml_prob")),
                        "hybrid_score": _safe_float(a.get("hybrid_score")),
                        "confidence": _safe_float(a.get("conf")),
                        "source_peak_intensity": _safe_float(a.get("source_peak_intensity")),
                        "source_peak_rel_intensity": _safe_float(a.get("source_peak_rel_intensity")),
                        "strict_export_intensity": _safe_float(export_intensity),
                        "strict_export_rel_intensity": _safe_float(export_rel_intensity),
                        "is_correct": matched,
                        "closest_aml_mz": _safe_float(closest_aml),
                        "mass_error_da": _safe_float(mass_error_da),
                        "mass_error_ppm": _safe_float(mass_error_ppm),
                    })

            strict_predicted = len(strict_selected)
            strict_has_true = strict_correct > 0
            retained_has_true = candidate_true > 0

            precision = (strict_correct / strict_predicted) if strict_predicted else None
            if precision is not None:
                precisions.append(precision)
                predicted_counts.append(strict_predicted)
                correct_counts.append(strict_correct)

            if RESULTS_OUTPUT.get("enabled"):
                strict_score_vals = [_safe_float(a.get("score")) for a in strict_selected]
                strict_ml_vals = [_safe_float(a.get("ml_prob")) for a in strict_selected]
                strict_hybrid_vals = [_safe_float(a.get("hybrid_score")) for a in strict_selected]
                strict_posterior_vals = [_safe_float(meta[a["candidate_id"]].get("peak_posterior")) for a in strict_selected]
                strict_decision_vals = [_safe_float(meta[a["candidate_id"]].get("decision_prob")) for a in strict_selected]
                rescued_count = sum(1 for a in strict_selected if meta[a["candidate_id"]].get("selection_stage") == "rescued")
                rescued_correct_count = sum(
                    1
                    for a in strict_selected
                    if meta[a["candidate_id"]].get("selection_stage") == "rescued"
                    and is_correct(_safe_float(a.get("best_exact_mz")), aml_mz)
                )
                strict_core_count = strict_predicted - rescued_count
                strict_core_correct_count = strict_correct - rescued_correct_count

                score_mean, score_max, score_min = _score_stats(strict_score_vals)
                ml_mean, _, _ = _score_stats(strict_ml_vals)
                hybrid_mean, _, _ = _score_stats(strict_hybrid_vals)
                posterior_mean, _, _ = _score_stats(strict_posterior_vals)
                decision_mean, _, _ = _score_stats(strict_decision_vals)

                entry_records.append({
                    "schema_version": RESULTS_OUTPUT.get("schema_version", "2.0"),
                    "run_id": run_id,
                    "entry_id": eid,
                    "parent_formula": parent_formula,
                    "parent_classes": parent_classes,
                    "precision": precision,
                    "strict_selected_count": strict_predicted,
                    "strict_correct_count": strict_correct,
                    "strict_core_count": strict_core_count,
                    "strict_core_correct_count": strict_core_correct_count,
                    "rescued_count": rescued_count,
                    "rescued_correct_count": rescued_correct_count,
                    "candidate_retained_count": len(retained_candidates),
                    "candidate_true_count": candidate_true,
                    "has_any_strict_true": strict_has_true,
                    "has_any_candidate_true": retained_has_true,
                    "score_mean": score_mean,
                    "score_max": score_max,
                    "score_min": score_min,
                    "ml_prob_mean": ml_mean,
                    "hybrid_score_mean": hybrid_mean,
                    "peak_posterior_mean": posterior_mean,
                    "decision_prob_mean": decision_mean,
                    "threshold_profile": "+".join(["default"] + [cls for cls in parent_classes if cls in STRICT_SELECTION.get("class_threshold_overrides", {})]),
                    "strict_hybrid_thr": STRICT_SELECTION["hybrid_thr"],
                    "strict_ml_prob_floor": STRICT_SELECTION["ml_prob_floor"],
                    "strict_peak_posterior_floor": STRICT_SELECTION["peak_posterior_floor"],
                    "rescue_max_expected_fdr": STRICT_SELECTION["rescue_max_expected_fdr"],
                    "strict_min_margin": STRICT_SELECTION["min_margin_to_runner_up"],
                    "candidate_top_per_peak": STRICT_SELECTION["candidate_top_per_peak"],
                    "tolerance_mz": TOL,
                    "status": "ok" if strict_predicted else "strict_none",
                })

            if RESULTS_OUTPUT.get("enabled") and RESULTS_OUTPUT.get("include_fake_spectra", False):
                run_dir = os.path.join(RESULTS_OUTPUT.get("out_dir", "validation_outputs_precision"), run_id)
                fake_dir = os.path.join(run_dir, RESULTS_OUTPUT.get("fake_spectra_dirname", "fake_spectra_strict"))
                path = export_fake_spectrum(
                    entry_id=eid,
                    parent_name=result["parent_name"],
                    parent_formula=parent_formula,
                    selected_assignments=strict_selected,
                    out_dir=fake_dir,
                )
                fake_spectrum_paths.append(path)

    arr = np.array(precisions, dtype=float) if precisions else np.array([], dtype=float)
    pred_arr = np.array(predicted_counts, dtype=float) if predicted_counts else np.array([], dtype=float)
    corr_arr = np.array(correct_counts, dtype=float) if correct_counts else np.array([], dtype=float)

    print("\n=== Precision-Oriented Hybrid Statistics ===")
    print(f"Entries processed: {len(entry_records)}")
    print(f"Entries with strict predictions: {len(arr)}")
    print(f"Entries with no strict predictions: {sum(1 for r in entry_records if r['strict_selected_count'] == 0)}")

    if len(arr):
        print("\n--- Strict precision (correct / strict_selected) ---")
        print(f"  Mean   : {arr.mean():.4f}")
        print(f"  Median : {np.median(arr):.4f}")
        print(f"  Stddev : {arr.std():.4f}")
        print(f"  Min    : {arr.min():.4f}")
        print(f"  Max    : {arr.max():.4f}")
        print(f"  25%    : {np.percentile(arr, 25):.4f}")
        print(f"  75%    : {np.percentile(arr, 75):.4f}")
        print("\n--- Strict fragment counts per spectrum ---")
        print(f"  Mean selected : {pred_arr.mean():.2f}")
        print(f"  Mean correct  : {corr_arr.mean():.2f}")
    else:
        print("\nNo strict predictions passed the precision gates.")

    import matplotlib.pyplot as plt

    plot_dir = None
    if RESULTS_OUTPUT.get("enabled"):
        plot_dir = os.path.join(RESULTS_OUTPUT.get("out_dir", "validation_outputs_precision"), run_id)
        os.makedirs(plot_dir, exist_ok=True)

    def _save_and_show(fig, filename):
        if plot_dir:
            fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
        plt.show()

    if len(arr):
        fig1 = plt.figure(figsize=(8, 5))
        plt.hist(arr, bins=15, color="steelblue", edgecolor="black", alpha=0.8)
        plt.title("Strict Precision Distribution Across Spectra")
        plt.xlabel("Precision (correct / strict_selected)")
        plt.ylabel("Number of spectra")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        _save_and_show(fig1, "precision_distribution.png")

        fig2 = plt.figure(figsize=(8, 5))
        plt.hist(pred_arr, bins=15, color="darkorange", edgecolor="black", alpha=0.8)
        plt.title("Distribution of Strict Selected Fragments per Spectrum")
        plt.xlabel("Number of strict selected fragments")
        plt.ylabel("Number of spectra")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        _save_and_show(fig2, "strict_fragment_count_distribution.png")

        fig3 = plt.figure(figsize=(8, 6))
        plt.scatter(pred_arr, arr, alpha=0.6, color="purple", edgecolor="black")
        plt.title("Strict Precision vs Number of Strict Selected Fragments")
        plt.xlabel("Number of strict selected fragments")
        plt.ylabel("Precision (correct / strict_selected)")
        if len(pred_arr) > 1 and np.ptp(pred_arr) > 0:
            z = np.polyfit(pred_arr, arr, 1)
            p = np.poly1d(z)
            xs = np.linspace(pred_arr.min(), pred_arr.max(), 200)
            plt.plot(xs, p(xs), "r--", linewidth=2, label="Trend line")
            plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        _save_and_show(fig3, "precision_vs_strict_fragment_count.png")

    files_written = {}
    if RESULTS_OUTPUT.get("enabled"):
        run_end = time.time()
        strict_pred_total = int(pred_arr.sum()) if len(pred_arr) else 0
        strict_corr_total = int(corr_arr.sum()) if len(corr_arr) else 0
        entry_has_true = sum(1 for r in entry_records if r["has_any_strict_true"])
        entry_candidate_has_true = sum(1 for r in entry_records if r["has_any_candidate_true"])
        rescued_total = int(sum(r.get("rescued_count", 0) for r in entry_records))
        rescued_correct_total = int(sum(r.get("rescued_correct_count", 0) for r in entry_records))
        strict_core_total = strict_pred_total - rescued_total
        strict_core_correct_total = strict_corr_total - rescued_correct_total

        run_summary = {
            "schema_version": RESULTS_OUTPUT.get("schema_version", "2.0"),
            "run_id": run_id,
            "mode": RESULTS_OUTPUT.get("mode", "precision_strict"),
            "started_at": datetime.fromtimestamp(run_start).isoformat(),
            "finished_at": datetime.fromtimestamp(run_end).isoformat(),
            "config": {
                "tol_mz": TOL,
                "min_rel_intensity": MIN_REL_INTENSITY,
                "strict_selection": STRICT_SELECTION,
                "final_decision_layer": {
                    k: v for k, v in FINAL_DECISION_LAYER.items() if k != "calibrator"
                },
                "use_multiplicative_hybrid": USE_MULTIPLICATIVE_HYBRID,
                "use_complementary_loss_filter": USE_COMPLEMENTARY_LOSS_FILTER,
                "use_mass_defect_filter": USE_MASS_DEFECT_FILTER,
                "notes": RESULTS_OUTPUT.get("notes", []),
            },
            "counts": {
                "entries_total": len(ENTRY_IDS),
                "entries_processed": len(entry_records),
                "entries_with_strict_predictions": len(arr),
                "entries_without_strict_predictions": sum(1 for r in entry_records if r["strict_selected_count"] == 0),
                "strict_fragments_selected": strict_pred_total,
                "strict_fragments_correct": strict_corr_total,
                "strict_core_fragments_selected": strict_core_total,
                "strict_core_fragments_correct": strict_core_correct_total,
                "strict_fragments_rescued": rescued_total,
                "strict_fragments_rescued_correct": rescued_correct_total,
                "candidate_fragments_retained": int(sum(r["candidate_retained_count"] for r in entry_records)),
                "entries_with_any_strict_true": entry_has_true,
                "entries_with_any_candidate_true": entry_candidate_has_true,
                "fake_spectra_written": len(fake_spectrum_paths),
            },
            "metrics": {
                "precision_macro_mean": float(arr.mean()) if len(arr) else None,
                "precision_macro_median": float(np.median(arr)) if len(arr) else None,
                "precision_micro": (strict_corr_total / strict_pred_total) if strict_pred_total else None,
                "precision_micro_strict_core": (strict_core_correct_total / strict_core_total) if strict_core_total else None,
                "precision_micro_rescued": (rescued_correct_total / rescued_total) if rescued_total else None,
                "entry_recall_any_strict_true": (entry_has_true / len(entry_records)) if entry_records else None,
                "entry_recall_any_candidate_true": (entry_candidate_has_true / len(entry_records)) if entry_records else None,
            },
        }

        files_written = save_validation_results(
            config=RESULTS_OUTPUT,
            run_id=run_id,
            entry_records=entry_records,
            fragment_records=fragment_records,
            run_summary=run_summary,
        )

    if files_written:
        print("\nSaved validation outputs:")
        for kind, path in files_written.items():
            print(f"  {kind}: {path}")
        if fake_spectrum_paths:
            print(f"  fake_spectra_dir: {os.path.dirname(fake_spectrum_paths[0])}")

    print("\nDone.")


if __name__ == "__main__":
    main()

