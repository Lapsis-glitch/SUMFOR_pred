# fragmentation_rules_info/phosphorus.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent
from .universal import ALPHA_CLEAVAGE_LOSSES


# ------------------------------------------------------------
# Phosphorus-specific neutral and radical losses
# ------------------------------------------------------------

PHOSPHORUS_LOSSES = [
    Formula({"P": 1, "O": 1}),          # PO•
    Formula({"P": 1, "O": 2}),          # PO2•
    Formula({"P": 1, "H": 3}),          # PH3 (phosphines)
]

PHOSPHORUS_RADICAL_LOSSES = [
    Formula({"P": 1}),                  # P•
]


# ------------------------------------------------------------
# Phosphorus cations
# ------------------------------------------------------------
PHOSPHORUS_CATIONS = [
    Formula({"P": 1}),                  # P+
    Formula({"P": 1, "O": 1}),          # PO+
    Formula({"P": 1, "O": 2}),          # PO2+
]

BETA_P_LOSSES = [
    Formula({"C": 2, "H": 5, "P": 1}),      # C2H5P
    Formula({"C": 2, "H": 4, "O": 1, "P": 1}),  # C2H4OP
    Formula({"C": 3, "H": 7, "P": 1}),      # C3H7P
]

def generate_beta_p(parent, fg):
    results = []
    if not fg.get("phosphorus"):
        return results

    for loss in BETA_P_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_P_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Phosphorus rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate phosphorus-specific EI fragments:
        - PO•, PO2•, PH3 losses
        - P• radical loss
        - C–P alpha cleavage
        - P+, PO+, PO2+ cations
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("phosphorus", False):
        return results

    # --------------------------------------------------------
    # 1. Neutral losses (PO, PO2, PH3)
    # --------------------------------------------------------
    for loss in PHOSPHORUS_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"phosphorus_loss_{loss.to_string()}"))

    # --------------------------------------------------------
    # 2. Radical loss (P•)
    # --------------------------------------------------------
    for loss in PHOSPHORUS_RADICAL_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"phosphorus_radical_loss_{loss.to_string()}"))

    # --------------------------------------------------------
    # 3. C–P alpha cleavage
    # --------------------------------------------------------
    for loss in ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None and "P" in frag.elements:
            results.append((frag, "phosphorus_alpha_cleavage"))

    # --------------------------------------------------------
    # 4. P-centered cations
    # --------------------------------------------------------
    for ion in PHOSPHORUS_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "phosphorus_cation"))

    results.extend(generate_beta_p(parent, fg))

    return results