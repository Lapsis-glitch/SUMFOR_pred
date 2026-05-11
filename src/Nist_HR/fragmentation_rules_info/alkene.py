# fragmentation_rules/alkene.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent


# ------------------------------------------------------------
# Allylic cations (common for alkenes)
# ------------------------------------------------------------
ALLYLIC_CATIONS = [
    Formula({"C": 3, "H": 5}),          # C3H5+
]


# ------------------------------------------------------------
# Alkene rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate alkene-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("alkene", False):
        return results

    # --- Allylic cation formation ---
    for ion in ALLYLIC_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "allylic_cation"))

    return results