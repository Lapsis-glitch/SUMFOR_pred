"""
runner.py

Batch pipeline:
1. Find matching XXX.jdx files in two folders
2. For each file:
    a. Load NIST EI spectrum (JDX)
    b. Extract parent formula
    c. Enumerate fragments (peak-driven)
    d. Assign peaks (peak-driven)
    e. Load high-resolution reference spectrum
    f. Score assignments
3. Print per-file scores + average score
"""

import os

from formula import Formula
from JDXParser import parse_jdx_mass

# NEW peak-driven modules
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine

# Old modules remain available but unused
from enumerator import FragmentEnumerator
from assignment_engine import AssignmentEngine

from utils import (
    write_assignments_csv,
    parse_reference_csv,
    convert_assignments,
)

from scoring import score_assignments


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

pathNIST = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_Ref/"
pathAML  = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/AML_renamed/"

# Peak-driven: ignore very weak peaks
MIN_REL_INTENSITY = 0.05   # keep peaks >= 5% of base peak


# ------------------------------------------------------------
# Compute COMMON_FILES once (used by main loop + tuning loop)
# ------------------------------------------------------------

nist_files = {f for f in os.listdir(pathNIST) if f.lower().endswith(".jdx")}
aml_files  = {f for f in os.listdir(pathAML)  if f.lower().endswith(".jdx")}

COMMON_FILES = sorted(nist_files.intersection(aml_files))

if not COMMON_FILES:
    raise RuntimeError("No matching .jdx files found in both folders.")

print(f"Found {len(COMMON_FILES)} matching files:")
for f in COMMON_FILES:
    print("  ", f)

print("\nRunning full pipeline...\n")

scores = []


# ------------------------------------------------------------
# Helper: run a single file (for tuning loop)
# ------------------------------------------------------------
def run_single_file(file):
    """
    Run the full peak-driven pipeline for a single NIST/AML pair.
    Returns the scoring result dict.
    """

    # Safety: skip files not present in both folders
    if file not in COMMON_FILES:
        return {"score": 0.0, "n_matches": 0, "n_total": 0, "matches": []}

    # 1. Load NIST EI spectrum
    spec = parse_jdx_mass(os.path.join(pathNIST, file))
    mz_ref = spec["mz"]
    intensity_ref = spec["intensity"]

    if not mz_ref:
        return {"score": 0.0, "n_matches": 0, "n_total": 0, "matches": []}

    # Compute relative intensities
    max_int = max(intensity_ref)
    rel_int = [i / max_int for i in intensity_ref]

    # Peak-driven: filter to high-intensity peaks
    nist_peaks_all = list(zip(mz_ref, intensity_ref, rel_int))
    nist_peaks_filtered = [
        (mz, I, rI) for (mz, I, rI) in nist_peaks_all
        if rI >= MIN_REL_INTENSITY
    ]

    if not nist_peaks_filtered:
        return {"score": 0.0, "n_matches": 0, "n_total": 0, "matches": []}

    mz_ref_filt = [mz for mz, _, _ in nist_peaks_filtered]
    intensity_ref_filt = [I for _, I, _ in nist_peaks_filtered]
    rel_int_filt = [rI for _, _, rI in nist_peaks_filtered]

    # 2. Extract parent formula
    parent_formula_str = spec["metadata"].get("MOLFORM")
    if not parent_formula_str:
        return {"score": 0.0, "n_matches": 0, "n_total": 0, "matches": []}

    parent_formula = Formula.from_string(parent_formula_str)

    # 3. Peak-driven fragment enumerator
    FRAG_DEPTH = 3

    rule_flags = {
        "neutral_losses": True,
        "double_neutral_losses": True,
        "common_cations": True,
        "alpha_cleavage": True,
        "rearrangements": True,
        "oxygen_adjacent": True,
        "hydrogen_transfer": True,
        "mclafferty": True,

        # functional groups
        "alcohol_rules": True,
        "carbonyl_rules": True,
        "aromatic_rules": True,
        "amine_rules": True,
        "ester_rules": True,
        "halogen_rules": True,
        "ether_rules": True,
        "alkene_rules": True,
    }

    peak_enum = PeakDrivenEnumerator(
        parent_formula,
        rule_flags=rule_flags,
        max_depth=FRAG_DEPTH,
        auto_detect_rules=True
    )

    # 4. Peak-driven assignment engine
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)

    nist_peaks = list(zip(mz_ref_filt, intensity_ref_filt))
    assignments = engine.assign_peaks(
        nist_peaks=nist_peaks,
        rel_intensities=rel_int_filt
    )

    assignments_dict = convert_assignments(assignments)

    # 5. Load high-resolution reference spectrum
    ref_peaks = parse_reference_csv(os.path.join(pathAML, file))

    # 6. Score assignments
    result = score_assignments(
        assignments_dict,
        ref_peaks,
        tol=0.0001,
        conf_threshold=0.49,
    )

    return result


# ------------------------------------------------------------
# Main loop over all matching files
# ------------------------------------------------------------

for file in COMMON_FILES:
    print("=" * 60)
    print(f"Processing {file}")
    print("=" * 60)

    # 1. Load NIST EI spectrum
    spec = parse_jdx_mass(os.path.join(pathNIST, file))
    mz_ref = spec["mz"]
    intensity_ref = spec["intensity"]

    if not mz_ref:
        print(f"Skipping {file}: empty spectrum.")
        continue

    # Compute relative intensities
    max_int = max(intensity_ref)
    rel_int = [i / max_int for i in intensity_ref]

    # Peak-driven: filter to high-intensity peaks
    nist_peaks_all = list(zip(mz_ref, intensity_ref, rel_int))
    nist_peaks_filtered = [
        (mz, I, rI) for (mz, I, rI) in nist_peaks_all
        if rI >= MIN_REL_INTENSITY
    ]

    if not nist_peaks_filtered:
        print(f"Skipping {file}: no peaks above {MIN_REL_INTENSITY*100:.1f}% rel. intensity.")
        continue

    mz_ref_filt = [mz for mz, _, _ in nist_peaks_filtered]
    intensity_ref_filt = [I for _, I, _ in nist_peaks_filtered]
    rel_int_filt = [rI for _, _, rI in nist_peaks_filtered]

    # 2. Extract parent formula
    parent_formula_str = spec["metadata"].get("MOLFORM")
    if not parent_formula_str:
        print(f"Skipping {file}: no MOLFORM in metadata.")
        continue

    parent_formula = Formula.from_string(parent_formula_str)
    print(f"Parent formula: {parent_formula.to_string()}")

    # 3. Peak-driven fragment enumerator
    FRAG_DEPTH = 3

    rule_flags = {
        "neutral_losses": True,
        "double_neutral_losses": True,
        "common_cations": True,
        "alpha_cleavage": True,
        "rearrangements": True,
        "oxygen_adjacent": True,
        "hydrogen_transfer": True,
        "mclafferty": True,

        # functional groups
        "alcohol_rules": True,
        "carbonyl_rules": True,
        "aromatic_rules": True,
        "amine_rules": True,
        "ester_rules": True,
        "halogen_rules": True,
        "ether_rules": True,
        "alkene_rules": True,
    }

    peak_enum = PeakDrivenEnumerator(
        parent_formula,
        rule_flags=rule_flags,
        max_depth=FRAG_DEPTH,
        auto_detect_rules=True
    )

    # 4. Peak-driven assignment engine
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)

    nist_peaks = list(zip(mz_ref_filt, intensity_ref_filt))
    assignments = engine.assign_peaks(
        nist_peaks=nist_peaks,
        rel_intensities=rel_int_filt
    )

    assignments_dict = convert_assignments(assignments)

    # Optional: show first multi-match
    for a in assignments:
        if len(a.candidate_formulas) > 1:
            print("MULTI:", a.nominal_mz, a.candidate_formulas)
            break

    # Save assignments for debugging
    out_csv = f"assignments_{file.replace('.jdx','')}.csv"
    write_assignments_csv(assignments_dict, out_csv)

    # 5. Load high-resolution reference spectrum
    ref_peaks = parse_reference_csv(os.path.join(pathAML, file))

    # 6. Score assignments
    result = score_assignments(
        assignments_dict,
        ref_peaks,
        tol=0.0001,
        conf_threshold=0.49,
    )

    print(f"Matched {result['n_matches']} out of {result['n_total']} guesses")
    print(f"Score = {result['score']:.3f}")

    scores.append(result["score"])

    # Optional: print matches
    for m in result["matches"]:
        print(f"Match: {m['best_formula']} at {m['best_exact_mz']:.5f} with conf: {m['conf']}")

    print("\n")


# ------------------------------------------------------------
# Final average score
# ------------------------------------------------------------

if scores:
    avg = sum(scores) / len(scores)
    print("=" * 60)
    print(f"Average score across {len(scores)} files: {avg:.3f}")
    print("=" * 60)
else:
    print("No scores computed.")