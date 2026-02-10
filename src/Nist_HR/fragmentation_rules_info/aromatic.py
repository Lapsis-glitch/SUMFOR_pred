# fragmentation_rules/aromatic.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Aromatic-specific ions and losses
# ------------------------------------------------------------
AROMATIC_TROPYLIUM_CATIONS = [
    Formula({"C": 7, "H": 7}),          # C7H7+
]

AROMATIC_PHENYL_CATIONS = [
    Formula({"C": 6, "H": 5}),          # C6H5+
]

AROMATIC_BENZYL_CATIONS = [
    Formula({"C": 7, "H": 7}),          # C7H7+ (benzyl/tropylium)
]

AROMATIC_RING_CONTRACTION_LOSSES = [
    Formula({"C": 2, "H": 2}),          # C2H2 loss
]


# ------------------------------------------------------------
# Aromatic rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate aromatic-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("aromatic", False):
        return results

    # --- Tropylium cation ---
    for ion in AROMATIC_TROPYLIUM_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "tropylium_cation"))

    # --- Phenyl cation ---
    for ion in AROMATIC_PHENYL_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "phenyl_cation"))

    # --- Benzyl cation ---
    for ion in AROMATIC_BENZYL_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "benzyl_cation"))

    # --- Ring contraction ---
    for loss in AROMATIC_RING_CONTRACTION_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"aromatic_ring_contraction_{loss.to_string()}"))

    return results