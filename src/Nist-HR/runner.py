"""
runner.py

Batch pipeline:
1. Find matching XXX.jdx files in two folders
2. For each file:
    a. Load NIST EI spectrum (JDX)
    b. Extract parent formula
    c. Enumerate fragments
    d. Assign peaks
    e. Load high-resolution reference spectrum
    f. Score assignments
3. Print per-file scores + average score
"""

import os

from formula import Formula
from enumerator import FragmentEnumerator
from assignment_engine import AssignmentEngine
from JDXParser import parse_jdx_mass
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine

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

# ------------------------------------------------------------
# Find matching .jdx files in both folders
# ------------------------------------------------------------

nist_files = {f for f in os.listdir(pathNIST) if f.lower().endswith(".jdx")}
aml_files  = {f for f in os.listdir(pathAML)  if f.lower().endswith(".jdx")}

common_files = sorted(nist_files.intersection(aml_files))

if not common_files:
    raise RuntimeError("No matching .jdx files found in both folders.")

print(f"Found {len(common_files)} matching files:")
for f in common_files:
    print("  ", f)

print("\nRunning full pipeline...\n")

scores = []


# ------------------------------------------------------------
# Main loop over all matching files
# ------------------------------------------------------------

for file in common_files:
    print("=" * 60)
    print(f"Processing {file}")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load NIST EI spectrum
    # --------------------------------------------------------
    spec = parse_jdx_mass(os.path.join(pathNIST, file))
    mz_ref = spec["mz"]
    intensity_ref = spec["intensity"]

    # --------------------------------------------------------
    # 2. Extract parent formula
    # --------------------------------------------------------
    parent_formula_str = spec["metadata"].get("MOLFORM")
    if not parent_formula_str:
        print(f"Skipping {file}: no MOLFORM in metadata.")
        continue

    parent_formula = Formula.from_string(parent_formula_str)
    print(f"Parent formula: {parent_formula.to_string()}")

    # --------------------------------------------------------
    # 3. Enumerate fragments
    # --------------------------------------------------------
    enumerator = FragmentEnumerator(parent_formula)
    fragments = enumerator.enumerate_subformulas()
    print(f"Generated {len(fragments)} fragments")

    # --------------------------------------------------------
    # 4. Assign peaks
    # --------------------------------------------------------
    engine = AssignmentEngine(parent_formula, fragments)
    nist_peaks = list(zip(mz_ref, intensity_ref))
    assignments = engine.assign_peaks(nist_peaks)

    assignments_dict = convert_assignments(assignments)

    # Optional: show first multi-match
    for a in assignments:
        if len(a.candidate_formulas) > 1:
            print("MULTI:", a.nominal_mz, a.candidate_formulas)
            break

    # Save assignments for debugging
    out_csv = f"assignments_{file.replace('.jdx','')}.csv"
    write_assignments_csv(assignments_dict, out_csv)

    # --------------------------------------------------------
    # 5. Load high-resolution reference spectrum
    # --------------------------------------------------------
    ref_peaks = parse_reference_csv(os.path.join(pathAML, file))

    # --------------------------------------------------------
    # 6. Score assignments
    # --------------------------------------------------------
    result = score_assignments(
        assignments_dict,
        ref_peaks,
        tol=0.0001,
        conf_threshold=0.78,
    )

    print(f"Matched {result['n_matches']} out of {result['n_total']} guesses")
    print(f"Score = {result['score']:.3f}")

    scores.append(result["score"])

    # Optional: print matches
    for m in result["matches"]:
        print(f"Match: {m['best_formula']} at {m['best_exact_mz']:.5f}")

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