# fragmentation_rules/universal.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Universal neutral losses
# ------------------------------------------------------------
NEUTRAL_LOSSES = [
    Formula({"H": 2, "O": 1}),          # H2O
    Formula({"C": 1, "H": 2}),          # CH2
    Formula({"C": 1, "H": 3}),          # CH3
    Formula({"C": 2, "H": 4}),          # C2H4
    Formula({"C": 2, "H": 5}),          # C2H5
    Formula({"C": 1, "O": 1}),          # CO
    Formula({"C": 1, "O": 2}),          # CO2
    Formula({"H": 1}),                  # H
    Formula({"H": 2}),                  # H2
]


# ------------------------------------------------------------
# Common EI cations
# ------------------------------------------------------------
COMMON_CATIONS = [
    Formula({"C": 1, "H": 3}),          # CH3+
    Formula({"C": 2, "H": 5}),          # C2H5+
    Formula({"C": 3, "H": 7}),          # C3H7+
    Formula({"C": 2, "H": 3, "O": 1}),  # C2H3O+
    Formula({"C": 1, "H": 1, "O": 1}),  # CHO+
    Formula({"C": 1, "O": 1}),          # CO+
    Formula({"H": 1}),                  # H+
    Formula({"O": 1, "H": 1}),          # OH+
]


# ------------------------------------------------------------
# Alpha-cleavage losses
# ------------------------------------------------------------
ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3}),
    Formula({"C": 1, "H": 2}),
    Formula({"C": 2, "H": 5}),
    Formula({"C": 2, "H": 4}),
]


# ------------------------------------------------------------
# NEW: Beta-cleavage losses
# ------------------------------------------------------------
BETA_CLEAVAGE_LOSSES = [
    Formula({"C": 2, "H": 5}),   # C2H5•
    Formula({"C": 2, "H": 4}),   # C2H4
    Formula({"C": 3, "H": 7}),   # C3H7•
]


def generate_beta_cleavage(parent: Formula, fg: dict):
    """
    Generic β-cleavage: triggered by any heteroatom.
    Produces stabilized alkyl cations and radicals.
    """
    results = []

    # Only activate if a heteroatom is present
    if not (
        fg.get("alcohol")
        or fg.get("ether")
        or fg.get("amine")
        or fg.get("halogen")
        or fg.get("sulfur")
        or fg.get("phosphorus")
    ):
        return results

    for loss in BETA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_cleavage_{loss.to_string()}"))

    return results


# ------------------------------------------------------------
# Generic rearrangement losses
# ------------------------------------------------------------
REARRANGEMENT_LOSSES = [
    Formula({"H": 2, "O": 1}),              # H2O
    Formula({"C": 1, "H": 4}),              # CH4
    Formula({"C": 1, "H": 2, "O": 1}),      # CH2O
]


# ------------------------------------------------------------
# Oxygen-adjacent losses
# ------------------------------------------------------------
OXYGEN_ADJACENT_LOSSES = [
    Formula({"C": 1, "H": 2, "O": 1}),      # CH2O
    Formula({"C": 1, "H": 4, "O": 1}),      # CH4O
]


# ------------------------------------------------------------
# Hydrogen transfer losses
# ------------------------------------------------------------
HYDROGEN_TRANSFER_LOSSES = [
    Formula({"H": 1}),
    Formula({"H": 2}),
    Formula({"H": 3}),
]


# ------------------------------------------------------------
# McLafferty rearrangements
# ------------------------------------------------------------
MCLAFFERTY_LOSSES = [
    Formula({"C": 2, "H": 4, "O": 1}),      # C2H4O
    Formula({"C": 3, "H": 6, "O": 1}),      # C3H6O
]


# ------------------------------------------------------------
# Universal rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Generate universal EI fragments.
    """
    results: List[Tuple[Formula, str]] = []
    elems = parent.elements

    # --- Neutral losses ---
    for loss in NEUTRAL_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"neutral_loss_{loss.to_string()}"))

    # --- Common cations ---
    for cat in COMMON_CATIONS:
        if subset_of_parent(cat, parent):
            results.append((cat, "common_cation"))

    # --- Alpha cleavage ---
    for loss in ALPHA_CLEAVAGE_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"alpha_cleavage_{loss.to_string()}"))

    # --- NEW: Beta cleavage ---
    results.extend(generate_beta_cleavage(parent, fg))

    # --- Rearrangements ---
    for loss in REARRANGEMENT_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"rearrangement_{loss.to_string()}"))

    # --- Oxygen-adjacent ---
    if elems.get("O", 0) > 0:
        for loss in OXYGEN_ADJACENT_LOSSES:
            frag = subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"oxygen_adjacent_{loss.to_string()}"))

    # --- Hydrogen transfer ---
    for loss in HYDROGEN_TRANSFER_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"hydrogen_transfer_{loss.to_string()}"))

    # --- McLafferty ---
    if fg.get("carbonyl", False):
        for loss in MCLAFFERTY_LOSSES:
            frag = subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"mclafferty_{loss.to_string()}"))

    return results