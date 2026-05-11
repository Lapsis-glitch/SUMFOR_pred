"""
Large_data.py

Data loader and single-entry pipeline runner for the merged NIST/AML dataset.

Responsibilities:
  - Loads the merged JSON dataset at import time (module-level).
  - Exposes ``ENTRY_IDS``: sorted list of all entry keys.
  - Exposes ``run_single_entry(entry_id)``: runs the full prediction pipeline
    for one compound and returns a result dict (or None on skip).

Pipeline stages per entry:
  1. Extract NIST spectrum and AML reference spectrum from the merged JSON.
  2. Parse parent molecular formula; skip unsupported elements.
  3. Filter NIST peaks by relative intensity threshold.
  4. Choose enumerator:
       - HybridEnumerator  (rules + BDE) if PubChem & BDE data are available.
       - PeakDrivenEnumerator  (rules only) as fallback.
  5. Assign fragments to NIST peaks via PeakDrivenAssignmentEngine.
  6. Apply LightGBM ML correction (adds ``ml_prob`` to each assignment).
  7. Compute hybrid score (multiplicative or linear, controlled by flag).
  8. Apply complementary-loss filter (controlled by flag).
  9. Keep only the top-K fragments by hybrid score.
  10. Deduplicate to one-best-per-peak (controlled by flag).

Precision-improvement flags (toggle on/off):
  - USE_BEST_PER_PEAK           — one fragment per NIST peak
  - USE_MULTIPLICATIVE_HYBRID   — √(physics × ML) instead of linear blend
  - USE_COMPLEMENTARY_LOSS_FILTER — remove chemically impossible neutral losses
"""

import json
import math

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from formula import Formula
from chemistry import exact_mass, dbe
from hybrid_enumerator import HybridEnumerator
from ml_correction_integration import apply_ml_correction
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from peak_driven_enumerator import PeakDrivenEnumerator
from utils import convert_assignments

# ── Configuration ────────────────────────────────────────────
MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"
MIN_REL_INTENSITY = 0.05      # minimum relative intensity to keep a NIST peak (0.05 = 5%, 0.10 = 10%)
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "Cl", "Br", "F", "I", "S", "P"}
FRAG_DEPTH = 5                # maximum recursive fragmentation depth
ALPHA = 0.4                   # hybrid score weight for physics  (1−α for ML)
TOP_K = 10                    # keep only the K highest-scoring fragments

# ── Precision-improvement flags ──────────────────────────────
# Set any flag to False to revert to the previous behaviour.

USE_BEST_PER_PEAK = False
"""Keep only the single highest-scoring fragment per nominal m/z.
Eliminates duplicate predictions from the same NIST peak."""

USE_MULTIPLICATIVE_HYBRID = True
"""Use ML-dominant hybrid score  ML^0.8 × physics^0.2  instead of
the linear blend  α·physics + (1−α)·ML.  ML is the stronger
discriminator, so it gets the dominant exponent; physics acts
as a tiebreaker to avoid inflating false positives."""

USE_COMPLEMENTARY_LOSS_FILTER = True
"""Discard fragments whose neutral loss (parent − fragment) has
negative atom counts or negative DBE, which is chemically
impossible."""

USE_MASS_DEFECT_FILTER = False
"""Discard fragments whose mass defect (fractional part of exact mass)
deviates too far from the expected range for their element class.
False positives have systematically larger mass defects (0.051 vs 0.040
for true positives)."""

MASS_DEFECT_LIMITS = {
    # (max_abs_defect) keyed by dominant element class
    "halogenated": 0.08,   # halogens pull defect wider
    "default":     0.05,   # pure CHO/N compounds
}

# All supported rule families (all enabled)
# Keys match the registry pack-level flag keys.
# Universal rules are always enabled (no flag key).
RULE_FLAGS = {
    "alcohol": True,
    "carbonyl": True,
    "aromatic": True,
    "amine": True,
    "ester": True,
    "halogen": True,
    "ether": True,
    "alkene": True,
    "sulfur": True,
    "phosphorus": True,
}


# ── Load merged dataset (runs once at import time) ───────────
with open(MERGED_PATH, "r") as _f:
    MERGED = json.load(_f)

ENTRY_IDS = sorted(MERGED.keys(), key=lambda x: int(x))


# ── Helpers ──────────────────────────────────────────────────

def _formula_has_only_allowed_elements(formula):
    """Return True if every element in *formula* is in ALLOWED_ELEMENTS."""
    return all(el in ALLOWED_ELEMENTS for el in formula.elements)


def _build_mol(entry):
    """
    Try to build an RDKit Mol (with explicit Hs) from PubChem data.

    Returns the Mol object or None.
    """
    pubchem = entry.get("pubchem")
    if not pubchem:
        return None

    smiles = pubchem.get("canonical_smiles") or pubchem.get("isomeric_smiles")
    inchi = pubchem.get("inchi")

    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    elif inchi:
        mol = Chem.MolFromInchi(inchi)

    if mol is not None:
        mol = Chem.AddHs(mol)
    return mol


def _extract_bde_data(entry):
    """
    Extract and filter BDE bond data from the entry.

    Returns a list of bond dicts (only those with non-null BDE), or None.
    """
    raw_bde = entry.get("bde_data")
    if not raw_bde or "bonds" not in raw_bde:
        return None
    bonds = [b for b in raw_bde["bonds"] if b["bde"] is not None]
    return bonds if bonds else None


# ── Main pipeline function ───────────────────────────────────

def _best_per_peak(assignments):
    """
    Keep only the highest hybrid-scoring fragment per nominal m/z.

    Assignments must already be sorted by hybrid_score descending.
    For each nominal_mz seen, only the first (best) entry is kept.
    """
    seen = set()
    out = []
    for a in assignments:
        nm = a.get("nominal_mz")
        if nm in seen:
            continue
        seen.add(nm)
        out.append(a)
    return out


def _filter_complementary_loss(assignments, parent_formula):
    """
    Remove fragments whose neutral loss (parent − fragment) is chemically
    impossible: negative atom counts or negative DBE.
    """
    parent_elems = parent_formula.elements
    filtered = []

    for a in assignments:
        frag_str = a.get("best_formula")
        if not frag_str:
            continue

        try:
            frag_f = Formula.from_string(frag_str)
        except Exception:
            continue

        # Compute neutral loss = parent − fragment
        loss_elems = {}
        valid = True
        for el, count in parent_elems.items():
            diff = count - frag_f.elements.get(el, 0)
            if diff < 0:
                valid = False
                break
            loss_elems[el] = diff

        # Check fragment doesn't contain elements absent from parent
        if valid:
            for el in frag_f.elements:
                if el not in parent_elems:
                    valid = False
                    break

        if not valid:
            continue

        # Check neutral-loss DBE is non-negative
        loss_formula = Formula(loss_elems)
        if dbe(loss_formula) < -0.5:      # small tolerance for rounding
            continue

        filtered.append(a)

    return filtered


def _filter_mass_defect(assignments, parent_formula):
    """
    Remove fragments whose mass defect (fractional part of exact mass)
    exceeds a class-dependent limit.

    Halogenated fragments naturally have wider mass defects so they
    get a more generous threshold.
    """
    parent_has_halogen = any(
        el in parent_formula.elements for el in ("Cl", "Br", "F", "I")
    )
    limit = MASS_DEFECT_LIMITS[
        "halogenated" if parent_has_halogen else "default"
    ]

    filtered = []
    for a in assignments:
        frag_mz = a.get("best_exact_mz")
        if frag_mz is None:
            continue
        defect = abs(frag_mz - round(frag_mz))
        if defect <= limit:
            filtered.append(a)
    return filtered


def run_single_entry(entry_id: str):
    """
    Run the full prediction pipeline for a single dataset entry.

    The AML reference peaks are returned for downstream evaluation but are
    **not** used during scoring — this ensures that validation is unbiased.

    Parameters
    ----------
    entry_id : str
        Key into the MERGED dict.

    Returns
    -------
    dict or None
        ``{"entry_id", "assignments", "nist_mz", "nist_int",
          "aml_mz", "aml_int", "parent_formula"}``
        Returns None if the entry cannot be processed (missing data,
        unsupported elements, no peaks after filtering, etc.).
    """
    entry = MERGED[entry_id]

    # ── 1. Extract AML (high-resolution reference) spectrum ──
    csv_entry = entry["csv_entry"]
    aml_mz = csv_entry["mz"]
    aml_int = csv_entry["intensities"]

    # ── 2. Extract NIST (low-resolution input) spectrum ──────
    if not entry["nist_matches"]:
        return None
    nist = entry["nist_matches"][0]
    nist_mz = nist["mz"]
    nist_int = nist["intensities"]
    if not nist_mz:
        return None

    # ── 3. Parent formula ────────────────────────────────────
    parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    if not parent_formula_str:
        return None

    parent_formula = Formula.from_string(parent_formula_str)
    if not _formula_has_only_allowed_elements(parent_formula):
        return None

    # ── 4. Compute relative intensities & filter low peaks ───
    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]

    nist_peaks_all = list(zip(nist_mz, nist_int, rel_int))
    nist_peaks_filtered = [
        (mz, I, rI) for mz, I, rI in nist_peaks_all
        if rI >= MIN_REL_INTENSITY
    ]
    if not nist_peaks_filtered:
        return None

    mz_filt = [mz for mz, _, _ in nist_peaks_filtered]
    int_filt = [I for _, I, _ in nist_peaks_filtered]
    rel_filt = [rI for _, _, rI in nist_peaks_filtered]

    # ── 5. Choose enumerator (hybrid vs rule-only) ───────────
    mol = _build_mol(entry)
    bde_data = _extract_bde_data(entry)

    if mol is not None and bde_data:
        peak_enum = HybridEnumerator(
            parent_formula, mol, bde_data,
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

    # ── 6. Assign fragments to filtered NIST peaks ───────────
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)
    assignments = engine.assign_peaks(
        nist_peaks=list(zip(mz_filt, int_filt)),
        rel_intensities=rel_filt,
    )

    # ── 7. ML correction (uses full NIST spectrum) ───────────
    apply_ml_correction(
        assignments,
        parent_name=csv_entry.get("name", ""),
        parent_formula_str=parent_formula_str,
        nist_mz=nist_mz,
        nist_int=rel_int,
    )

    # ── 8. Convert to dicts & compute hybrid score ───────────
    assignments_dict = convert_assignments(assignments)

    for a in assignments_dict:
        physics = a.get("score", 0.0)
        ml = a.get("ml_prob", 0.0)

        if USE_MULTIPLICATIVE_HYBRID:
            # ML-dominant hybrid: ML is the stronger discriminator
            a["hybrid_score"] = (max(ml, 1e-9) ** 0.8) * (max(physics, 1e-9) ** 0.2)
        else:
            # Legacy: linear blend with ML override
            a["hybrid_score"] = ALPHA * physics + (1 - ALPHA) * ml
            if physics < 0.1 and ml >= 0.8:
                a["hybrid_score"] = ml

    # ── 8b. Complementary-loss filter ────────────────────────
    if USE_COMPLEMENTARY_LOSS_FILTER:
        assignments_dict = _filter_complementary_loss(
            assignments_dict, parent_formula
        )

    # ── 8c. Mass-defect filter ────────────────────────────────
    if USE_MASS_DEFECT_FILTER:
        assignments_dict = _filter_mass_defect(
            assignments_dict, parent_formula
        )

    # ── 9. Top-K filter ──────────────────────────────────────
    assignments_dict.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    assignments_dict = assignments_dict[:TOP_K]

    # ── 9b. One-best-per-peak deduplication ──────────────────
    if USE_BEST_PER_PEAK:
        assignments_dict = _best_per_peak(assignments_dict)

    return {
        "entry_id": entry_id,
        "assignments": assignments_dict,
        "nist_mz": nist_mz,
        "nist_int": nist_int,
        "aml_mz": aml_mz,
        "aml_int": aml_int,
        "parent_formula": parent_formula_str,
    }


# ── Standalone smoke-test ────────────────────────────────────

if __name__ == "__main__":
    processed = 0
    for eid in ENTRY_IDS:
        result = run_single_entry(eid)
        if result is not None:
            processed += 1
            print(f"Processed entry {eid}")
    print(f"\nProcessed {processed} / {len(ENTRY_IDS)} entries.")

