# clean_merged_dataset.py

import json
import re
from formula import Formula

MERGED_PATH = "/mnt/d/Leco/merged.json"
OUT_PATH    = "/mnt/d/Leco/merged_clean_SIP.json"

# Allowed elements for your fragmentation engine
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "Cl", "Br", "F", "I", "S", "P"}

def formula_has_only_allowed_elements(formula):
    """Return False if formula contains unsupported elements."""
    for el in formula.elements:
        if el not in ALLOWED_ELEMENTS:
            return False
    return True

def formula_is_valid(formula_str):
    """Reject formulas with weird characters or isotopic labels."""
    return bool(re.fullmatch(r"[A-Za-z0-9]+", formula_str))


def main():
    with open(MERGED_PATH, "r") as f:
        merged = json.load(f)

    cleaned = {}
    removed = 0

    for entry_id, entry in merged.items():

        csv_entry = entry.get("csv_entry", {})
        aml_mz = csv_entry.get("mz", [])
        aml_int = csv_entry.get("intensities", [])

        # Skip empty AML spectra
        if not aml_mz or not aml_int:
            removed += 1
            continue

        # Skip missing NIST matches
        nist_matches = entry.get("nist_matches")
        if not nist_matches:
            removed += 1
            continue

        nist = nist_matches[0]
        nist_mz = nist.get("mz", [])
        nist_int = nist.get("intensities", [])

        # Skip empty NIST spectra
        if not nist_mz or not nist_int:
            removed += 1
            continue

        # Parent formula
        parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
        if not parent_formula_str:
            removed += 1
            continue

        # Skip malformed formulas
        if not formula_is_valid(parent_formula_str):
            removed += 1
            continue

        try:
            parent_formula = Formula.from_string(parent_formula_str)
        except Exception:
            removed += 1
            continue

        # Skip unsupported elements
        if not formula_has_only_allowed_elements(parent_formula):
            removed += 1
            continue

        # Extract InChIKeys safely
        inchi_keys = [m.get("inchikey") for m in nist_matches if m.get("inchikey")]

        # Passed all checks → keep entry
        cleaned[entry_id] = {
            "csv_entry": csv_entry,
            "nist_matches": nist_matches,
            "inchi_keys": inchi_keys
        }

    # Save cleaned dataset
    with open(OUT_PATH, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Original entries: {len(merged)}")
    print(f"Removed entries:  {removed}")
    print(f"Cleaned entries:  {len(cleaned)}")
    print(f"Saved cleaned dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()