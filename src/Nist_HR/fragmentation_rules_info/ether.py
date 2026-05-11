# fragmentation_rules/ether.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent

# NOTE: BETA_O_LOSSES and generate_beta_o moved to base.py (shared with alcohol)
# and called from universal.py to avoid duplication.


# ------------------------------------------------------------
# Ether-specific alpha-cleavage losses
# ------------------------------------------------------------
ETHER_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]


# ------------------------------------------------------------
# Ether rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate ether-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("ether", False):
        return results

    # --- Alpha cleavage next to oxygen ---
    for loss in ETHER_ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"ether_alpha_cleavage_{loss.to_string()}"))


    return results