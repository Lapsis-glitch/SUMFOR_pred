# fragmentation_rules/halogen.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Halogen-specific neutral losses (HX)
# ------------------------------------------------------------
HALOGEN_LOSSES = [
    Formula({"H": 1, "F": 1}),          # HF
    Formula({"H": 1, "Cl": 1}),         # HCl
    Formula({"H": 1, "Br": 1}),         # HBr
    Formula({"H": 1, "I": 1}),          # HI
]

# ------------------------------------------------------------
# Halogen radical losses (X•)
# ------------------------------------------------------------
HALOGEN_RADICAL_LOSSES = [
    Formula({"F": 1}),                  # F•
    Formula({"Cl": 1}),                 # Cl•
    Formula({"Br": 1}),                 # Br•
    Formula({"I": 1}),                  # I•
]

# ------------------------------------------------------------
# Halogen cations (X+)
# ------------------------------------------------------------
HALOGEN_CATIONS = [
    Formula({"F": 1}),                  # F+
    Formula({"Cl": 1}),                 # Cl+
    Formula({"Br": 1}),                 # Br+
    Formula({"I": 1}),                  # I+
]

BETA_X_LOSSES = [
    Formula({"C": 1, "H": 2, "F": 1}),
    Formula({"C": 1, "H": 2, "Cl": 1}),
    Formula({"C": 1, "H": 2, "Br": 1}),
    Formula({"C": 1, "H": 2, "I": 1}),
]

def generate_beta_x(parent, fg):
    results = []
    if not fg.get("halogen"):
        return results

    for loss in BETA_X_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_halogen_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Halogen rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate halogen-specific EI fragments:
        - HX losses
        - X• radical losses
        - C–X alpha cleavage (approximated via radical subtraction)
        - Halogen cations
    """
    results: List[Tuple[Formula, str]] = []

    if not fg.get("halogen", False):
        return results

    # --------------------------------------------------------
    # 1. HX losses (HF, HCl, HBr, HI)
    # --------------------------------------------------------
    for loss in HALOGEN_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"halogen_loss_{loss.to_string()}"))

    # --------------------------------------------------------
    # 2. X• radical losses (F•, Cl•, Br•, I•)
    # --------------------------------------------------------
    for loss in HALOGEN_RADICAL_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"halogen_radical_loss_{loss.to_string()}"))

    # --------------------------------------------------------
    # 3. C–X alpha cleavage
    # --------------------------------------------------------
    # Approximated by subtracting X• (same as radical loss)
    # but labeled separately for interpretability.
    for loss in HALOGEN_RADICAL_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"halogen_alpha_cleavage_{loss.to_string()}"))

    # --------------------------------------------------------
    # 4. Halogen cations (X+)
    # --------------------------------------------------------
    for ion in HALOGEN_CATIONS:
        if subset_of_parent(ion, parent):
            results.append((ion, "halogen_cation"))

    results.extend(generate_beta_x(parent, fg))

    return results