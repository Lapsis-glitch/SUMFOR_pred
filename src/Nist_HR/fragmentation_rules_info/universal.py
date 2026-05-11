# fragmentation_rules/universal.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent, generate_beta_o


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
# Alpha-cleavage losses (C5: extended to C4)
# ------------------------------------------------------------
ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3}),
    Formula({"C": 1, "H": 2}),
    Formula({"C": 2, "H": 5}),
    Formula({"C": 2, "H": 4}),
    # C5: Larger alkyl series for long-chain aliphatics
    Formula({"C": 3, "H": 6}),          # C3H6
    Formula({"C": 3, "H": 7}),          # C3H7•
    Formula({"C": 4, "H": 8}),          # C4H8
    Formula({"C": 4, "H": 9}),          # C4H9•
]


# ------------------------------------------------------------
# Beta-cleavage losses
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
# C10: Double / consecutive neutral losses
# Pre-computed at depth 1 to avoid depth penalty from recursion
# ------------------------------------------------------------
DOUBLE_NEUTRAL_LOSSES = [
    (Formula({"C": 1, "H": 2, "O": 2}), "double_loss_H2O+CO"),       # H2O + CO  (46 Da)
    (Formula({"H": 4, "O": 2}),          "double_loss_2xH2O"),        # 2× H2O   (36 Da)
    (Formula({"C": 2, "O": 2}),          "double_loss_2xCO"),         # 2× CO    (56 Da)
    (Formula({"C": 1, "H": 2, "O": 3}), "double_loss_CO2+H2O"),      # CO2 + H2O (62 Da)
    (Formula({"C": 2, "H": 2, "O": 1}), "double_loss_CO+CH2"),       # CO + CH2  (42 Da)
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

    # --- Beta cleavage ---
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

    # --- C10: Double neutral losses ---
    for loss, label in DOUBLE_NEUTRAL_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, label))

    # --- A3: Shared beta-O cleavage (alcohol/ether) ---
    # Called here (once) to avoid duplication between alcohol.py and ether.py
    results.extend(generate_beta_o(parent, fg))

    return results