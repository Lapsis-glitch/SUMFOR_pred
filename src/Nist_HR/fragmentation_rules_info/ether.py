# fragmentation_rules/ether.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Ether-specific alpha-cleavage losses
# ------------------------------------------------------------
ETHER_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
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

    results.extend(generate_beta_o(parent, fg))

    return results