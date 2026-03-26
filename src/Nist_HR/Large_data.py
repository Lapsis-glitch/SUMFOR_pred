# runner_merged.py

import json

from formula import Formula
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from utils import convert_assignments
from hybrid_enumerator import HybridEnumerator
from rdkit import Chem

# ML correction
from ml_correction_integration import apply_ml_correction

MIN_REL_INTENSITY = 0.05
# MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP.json"
MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"

# Allowed elements
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "Cl", "Br", "F", "I", "S", "P"}


def formula_has_only_allowed_elements(formula):
    return all(el in ALLOWED_ELEMENTS for el in formula.elements)


# Load merged dataset
with open(MERGED_PATH, "r") as f:
    MERGED = json.load(f)

ENTRY_IDS = sorted(MERGED.keys(), key=lambda x: int(x))


def run_single_entry(entry_id: str):
    """
    Runs the full NIST → fragments → physics → ML → hybrid pipeline.
    DOES NOT use AML for scoring.
    Returns:
        {
            "assignments": [...],
            "nist_mz": [...],
            "nist_int": [...],
            "aml_mz": [...],
            "aml_int": [...],
            "parent_formula": "...",
            "entry_id": ...
        }
    """
    entry = MERGED[entry_id]

    csv_entry = entry["csv_entry"]
    aml_mz = csv_entry["mz"]
    aml_int = csv_entry["intensities"]

    # NIST spectrum
    if not entry["nist_matches"]:
        return None

    nist = entry["nist_matches"][0]
    nist_mz = nist["mz"]
    nist_int = nist["intensities"]

    if not nist_mz:
        return None

    # Parent formula
    parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    if not parent_formula_str:
        return None

    parent_formula = Formula.from_string(parent_formula_str)

    # Skip unsupported elements
    if not formula_has_only_allowed_elements(parent_formula):
        print(f"Skipping entry {entry_id}: unsupported elements in {parent_formula_str}")
        return None

    # Relative intensities (FULL NIST, for ML)
    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]

    # Filter NIST peaks (for physics engine)
    nist_peaks_all = list(zip(nist_mz, nist_int, rel_int))
    nist_peaks_filtered = [
        (mz, I, rI) for (mz, I, rI) in nist_peaks_all
        if rI >= MIN_REL_INTENSITY
    ]

    if not nist_peaks_filtered:
        return None

    mz_ref_filt = [mz for mz, _, _ in nist_peaks_filtered]
    intensity_ref_filt = [I for _, I, _ in nist_peaks_filtered]
    rel_int_filt = [rI for _, _, rI in nist_peaks_filtered]

    # Fragment rules
    FRAG_DEPTH = 5
    rule_flags = {
        "neutral_losses": True,
        "double_neutral_losses": True,
        "common_cations": True,
        "alpha_cleavage": True,
        "rearrangements": True,
        "oxygen_adjacent": True,
        "hydrogen_transfer": True,
        "mclafferty": True,
        "alcohol_rules": True,
        "carbonyl_rules": True,
        "aromatic_rules": True,
        "amine_rules": True,
        "ester_rules": True,
        "halogen_rules": True,
        "ether_rules": True,
        "alkene_rules": True,
    }

    #OLD implementation with just rules
    # peak_enum = PeakDrivenEnumerator(
    #     parent_formula,
    #     rule_flags=rule_flags,
    #     max_depth=FRAG_DEPTH,
    #     auto_detect_rules=True,
    # )

    #New hybrid implementation that merges rules + BDE-based fragments
    # -----------------------------
    # Hybrid or fallback enumerator
    # -----------------------------
    pubchem = entry.get("pubchem")
    # bde_data = entry.get("bde_data") or entry.get("bde")  # support both names
    raw_bde = entry.get("bde_data")
    bde_data = None

    if raw_bde and "bonds" in raw_bde:
        # Filter out null BDEs
        bde_data = [b for b in raw_bde["bonds"] if b["bde"] is not None]

    use_hybrid = False
    mol = None

    # Try to build RDKit mol if possible
    if pubchem:
        smiles = pubchem.get("canonical_smiles") or pubchem.get("isomeric_smiles")
        inchi = pubchem.get("inchi")

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        elif inchi:
            mol = Chem.MolFromInchi(inchi)

        if mol is not None:
            mol = Chem.AddHs(mol)

    # Decide whether hybrid is possible
    if mol is not None and bde_data:
        use_hybrid = True


    # Choose enumerator
    if use_hybrid:
        peak_enum = HybridEnumerator(
            parent_formula,
            mol,
            bde_data,
            rule_flags=rule_flags,
            max_depth=FRAG_DEPTH,
            bde_threshold=120.0,
            bde_softness=50.0,
        )
    else:
        # Fallback: rule-based only
        peak_enum = PeakDrivenEnumerator(
            parent_formula,
            rule_flags=rule_flags,
            max_depth=FRAG_DEPTH,
            auto_detect_rules=True,
        )

    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)

    # Assign peaks (physics uses filtered NIST)
    nist_peaks = list(zip(mz_ref_filt, intensity_ref_filt))
    assignments = engine.assign_peaks(
        nist_peaks=nist_peaks,
        rel_intensities=rel_int_filt,
    )

    # Apply ML correction (ML sees FULL NIST + REL INT)
    apply_ml_correction(
        assignments,
        parent_name=csv_entry.get("name", ""),
        parent_formula_str=parent_formula_str,
        nist_mz=nist_mz,   # full list
        nist_int=rel_int,  # relative intensities
    )

    # Convert to dicts
    assignments_dict = convert_assignments(assignments)

    # Add hybrid score
    ALPHA = 0.4  # weight for physics
    for a in assignments_dict:
        base = a.get("score", 0.0)
        mlp = a.get("ml_prob", 0.0)
        a["hybrid_score"] = ALPHA * base + (1 - ALPHA) * mlp

        # If physics is very low but ML is confident, override hybrid score
        if base < 0.1 and mlp >= 0.8:
            a["hybrid_score"] = mlp  # trust ML

    # ------------------------------------------------------------
    # Top-K fragment cap (keep only the highest-scoring fragments)
    # ------------------------------------------------------------
    TOP_K = 20  # or 15 if you want to be stricter

    # Sort assignments by hybrid score descending
    assignments_dict.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    # Keep only the top K
    assignments_dict = assignments_dict[:TOP_K]

    return {
        "entry_id": entry_id,
        "assignments": assignments_dict,
        "nist_mz": nist_mz,
        "nist_int": nist_int,
        "aml_mz": aml_mz,
        "aml_int": aml_int,
        "parent_formula": parent_formula_str,
    }


if __name__ == "__main__":
    count = 0
    for eid in ENTRY_IDS:
        result = run_single_entry(eid)
        if result is not None:
            count += 1
            print(f"Processed entry {eid}")

    print(f"\nProcessed {count} entries.")