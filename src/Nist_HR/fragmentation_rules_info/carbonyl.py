# fragmentation_rules/carbonyl.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Carbonyl-specific ions and losses
# ------------------------------------------------------------
ACYLIUM_IONS = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3CO+
    Formula({"C": 2, "H": 5, "O": 1}),  # C2H5CO+
]

CARBONYL_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 1}),          # CH•
    Formula({"C": 1, "H": 3}),          # CH3•
]


# ------------------------------------------------------------
# Carbonyl rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate carbonyl-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("carbonyl", False):
        return results

    # --- Acylium ions ---
    for ion in ACYLIUM_IONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "acylium_ion"))

    # --- Alpha cleavage next to carbonyl ---
    for loss in CARBONYL_ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"carbonyl_alpha_cleavage_{loss.to_string()}"))

    return results