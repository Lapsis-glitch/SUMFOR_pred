# fragmentation_rules/alcohol.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Alcohol-specific losses
# ------------------------------------------------------------
ALCOHOL_BETA_CLEAVAGE_LOSSES = [
    Formula({"C": 2, "H": 5, "O": 1}),  # generic C2H5O loss
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]

ALCOHOL_DEHYDRATION_LOSSES = [
    Formula({"H": 2, "O": 1}),          # H2O
]

ALCOHOL_GAMMA_H_SHIFT_LOSSES = [
    Formula({"C": 1, "H": 2}),          # CH2
    Formula({"C": 1, "H": 3}),          # CH3
]

BETA_O_LOSSES = [
    Formula({"C": 1, "H": 2, "O": 1}),   # CH2O
    Formula({"C": 2, "H": 4, "O": 1}),   # C2H4O
    Formula({"C": 2, "H": 6, "O": 1}),   # C2H6O
]

def generate_beta_o(parent, fg):
    results = []
    if not (fg.get("alcohol") or fg.get("ether")):
        return results

    for loss in BETA_O_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_O_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Alcohol rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate alcohol-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("alcohol", False):
        return results

    # --- Beta cleavage ---
    for loss in ALCOHOL_BETA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"alcohol_beta_cleavage_{loss.to_string()}"))

    # --- Dehydration ---
    for loss in ALCOHOL_DEHYDRATION_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"alcohol_dehydration_{loss.to_string()}"))

    # --- Gamma hydrogen shift ---
    for loss in ALCOHOL_GAMMA_H_SHIFT_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"alcohol_gamma_H_shift_{loss.to_string()}"))

    results.extend(generate_beta_o(parent, fg))

    return results