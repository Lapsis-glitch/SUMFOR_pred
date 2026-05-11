# fragmentation_rules/ester.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent


# ------------------------------------------------------------
# Ester-specific ions and losses
# ------------------------------------------------------------
ESTER_ACYLIUM_CATIONS = [
    Formula({"C": 2, "H": 3, "O": 1}),  # C2H3O+
    Formula({"C": 3, "H": 5, "O": 1}),  # C3H5O+
]

ESTER_ALKOXY_LOSSES = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]


# ------------------------------------------------------------
# Ester rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate ester-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("ester", False):
        return results

    # --- Acylium ions ---
    for ion in ESTER_ACYLIUM_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "ester_acylium_cation"))

    # --- Alkoxy cleavage ---
    for loss in ESTER_ALKOXY_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"ester_alkoxy_loss_{loss.to_string()}"))

    return results