# fragmentation_rules/sulfur.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent
from .universal import ALPHA_CLEAVAGE_LOSSES


# ------------------------------------------------------------
# Sulfur-specific neutral losses
# ------------------------------------------------------------
H2S_LOSS = Formula({"H": 2, "S": 1})     # thiols, thioethers
SO2_LOSS = Formula({"S": 1, "O": 2})     # sulfones, sulfonates, sulfonamides

BETA_S_LOSSES = [
    Formula({"C": 2, "H": 6, "S": 1}),   # C2H6S
    Formula({"C": 3, "H": 8, "S": 1}),   # C3H8S
]

# C9: Thiophene / thiophenyl cations (sulfur aromatics)
SULFUR_AROMATIC_CATIONS = [
    Formula({"C": 4, "H": 3, "S": 1}),   # C4H3S+  (83 Da) — thiophene cation
    Formula({"C": 1, "H": 1, "S": 1}),   # CHS+    (45 Da)
]

def generate_beta_s(parent, fg):
    results = []
    if not fg.get("sulfur"):
        return results

    for loss in BETA_S_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_S_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Sulfur rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate sulfur-specific EI fragments:
        - C–S alpha cleavage
        - H2S loss
        - SO2 loss
    """
    results: List[Tuple[Formula, str]] = []

    # If no sulfur, skip all rules
    if not fg.get("sulfur", False):
        return results

    # --------------------------------------------------------
    # Rule 1: C–S alpha cleavage
    # --------------------------------------------------------
    # We approximate alpha-cleavage by subtracting small alkyl radicals,
    # same approach as oxygen alpha-cleavage in your original engine.
    for loss in ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            # Keep only fragments that still contain sulfur
            if "S" in frag.elements:
                results.append((frag, "sulfur_alpha"))

    # --------------------------------------------------------
    # Rule 2: H2S loss (thiols, thioethers)
    # --------------------------------------------------------
    if fg.get("thiol_or_thioether", False):
        frag = subtract(parent, H2S_LOSS)
        if frag is not None:
            results.append((frag, "sulfur_H2S_loss"))

    # --------------------------------------------------------
    # Rule 3: SO2 loss (sulfones, sulfonates, sulfonamides)
    # --------------------------------------------------------
    if fg.get("sulfonyl", False):
        frag = subtract(parent, SO2_LOSS)
        if frag is not None:
            results.append((frag, "sulfur_SO2_loss"))

    # --------------------------------------------------------
    # Rule 4: Beta-S losses
    # --------------------------------------------------------
    results.extend(generate_beta_s(parent, fg))

    # --------------------------------------------------------
    # Rule 5: C9 — Sulfur-aromatic cations (thiophene, CHS+)
    # Gate: sulfur + aromatic, or DBE >= 2 (catches 5-membered
    #       S-heterocycles like thiophene which has DBE = 3)
    # --------------------------------------------------------
    from chemistry import dbe as _dbe
    if fg.get("aromatic", False) or _dbe(parent) >= 2:
        for ion in SULFUR_AROMATIC_CATIONS:
            if subset_of_parent(ion, parent):
                results.append((ion, "sulfur_aromatic_cation"))

    return results