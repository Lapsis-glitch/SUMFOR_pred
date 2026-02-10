# fragmentation_rules/amine.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract,subset_of_parent

# ------------------------------------------------------------
# Amine-specific ions and losses
# ------------------------------------------------------------
IMINIUM_IONS = [
    Formula({"C": 1, "H": 4, "N": 1}),  # CH4N+
    Formula({"C": 2, "H": 6, "N": 1}),  # C2H6N+
]

AMINE_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3}),          # CH3•
]

BETA_N_LOSSES = [
    Formula({"C": 2, "H": 6, "N": 1}),   # C2H6N
    Formula({"C": 3, "H": 8, "N": 1}),   # C3H8N
]

def generate_beta_n(parent, fg):
    results = []
    if not fg.get("amine"):
        return results

    for loss in BETA_N_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_N_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Amine rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate amine-specific EI fragments.
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("amine", False):
        return results

    # --- Iminium ions ---
    for ion in IMINIUM_IONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "iminium_ion"))

    # --- Alpha cleavage next to nitrogen ---
    for loss in AMINE_ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"amine_alpha_cleavage_{loss.to_string()}"))

    results.extend(generate_beta_n(parent, fg))

    return results