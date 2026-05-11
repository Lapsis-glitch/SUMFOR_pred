# fragmentation_rules/alcohol.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent


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

# NOTE: BETA_O_LOSSES and generate_beta_o moved to base.py (shared with ether)
# and called from universal.py to avoid duplication.


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


    return results