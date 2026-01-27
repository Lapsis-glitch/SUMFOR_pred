"""
fragmentation_rules.py

Defines rule-based fragment generation for EI mass spectrometry.

This module contains:
- universal combinatorial logic:
  - subtracting neutral losses
  - generating common cations
  - alpha-cleavage patterns
  - rearrangement losses
  - hydrogen transfer
  - McLafferty rearrangements
- functional-group–specific rules:
  - alcohols
  - carbonyls
  - aromatics
  - amines
  - esters
  - halogens
  - ethers
  - alkenes

Each generated fragment is returned as a tuple:
    (Formula, rule_source)

Formula objects remain immutable.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from formula import Formula
from chemistry import exact_mass, dbe


# ------------------------------------------------------------
# Universal neutral losses (apply to most organic molecules)
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
# Common EI cations (ubiquitous across organic MS)
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
# Alpha-cleavage losses (universal next to heteroatoms)
# ------------------------------------------------------------
ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3}),
    Formula({"C": 1, "H": 2}),
    Formula({"C": 2, "H": 5}),
    Formula({"C": 2, "H": 4}),
]

# ------------------------------------------------------------
# Generic rearrangement losses
# ------------------------------------------------------------
REARRANGEMENT_LOSSES = [
    Formula({"H": 2, "O": 1}),              # H2O
    Formula({"C": 1, "H": 4}),              # CH4
    Formula({"C": 1, "H": 2, "O": 1}),      # CH2O
]

# ------------------------------------------------------------
# Oxygen-adjacent losses (universal for oxygenates)
# ------------------------------------------------------------
OXYGEN_ADJACENT_LOSSES = [
    Formula({"C": 1, "H": 2, "O": 1}),      # CH2O
    Formula({"C": 1, "H": 4, "O": 1}),      # CH4O
]

# ------------------------------------------------------------
# Hydrogen transfer losses (very general)
# ------------------------------------------------------------
HYDROGEN_TRANSFER_LOSSES = [
    Formula({"H": 1}),
    Formula({"H": 2}),
    Formula({"H": 3}),
]

# ------------------------------------------------------------
# McLafferty rearrangements (general carbonyl chemistry)
# ------------------------------------------------------------
MCLAFFERTY_LOSSES = [
    Formula({"C": 2, "H": 4, "O": 1}),      # C2H4O
    Formula({"C": 3, "H": 6, "O": 1}),      # C3H6O
]


# ============================================================
# Functional-Group–Specific Rule Families
# ============================================================

# --- Alcohols ---
ALCOHOL_BETA_CLEAVAGE_LOSSES = [
    Formula({"C": 2, "H": 5, "O": 1}),  # generic C2H5O loss
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]

ALCOHOL_DEHYDRATION_LOSSES = [
    Formula({"H": 2, "O": 1}),          # H2O
]

ALCOHOL_GAMMA_H_SHIFT_LOSSES = [
    Formula({"C": 1, "H": 2}),          # CH2
    Formula({"C": 1, "H": 3}),          # CH3
]

# --- Carbonyls ---
ACYLIUM_IONS = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3CO+
    Formula({"C": 2, "H": 5, "O": 1}),  # C2H5CO+
]

CARBONYL_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 1}),          # CH•
    Formula({"C": 1, "H": 3}),          # CH3•
]

# --- Aromatics ---
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

# --- Amines ---
IMINIUM_IONS = [
    Formula({"C": 1, "H": 4, "N": 1}),  # CH4N+
    Formula({"C": 2, "H": 6, "N": 1}),  # C2H6N+
]

AMINE_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3}),          # CH3•
]

# --- Esters ---
ESTER_ACYLIUM_CATIONS = [
    Formula({"C": 2, "H": 3, "O": 1}),  # C2H3O+
    Formula({"C": 3, "H": 5, "O": 1}),  # C3H5O+
]

ESTER_ALKOXY_LOSSES = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]

# --- Halogens ---
HALOGEN_LOSSES = [
    Formula({"H": 1, "Cl": 1}),         # HCl
    Formula({"H": 1, "Br": 1}),         # HBr
]

HALOGEN_CATIONS = [
    Formula({"Cl": 1}),                 # Cl+
    Formula({"Br": 1}),                 # Br+
]

# --- Ethers ---
ETHER_ALPHA_CLEAVAGE_LOSSES = [
    Formula({"C": 1, "H": 3, "O": 1}),  # CH3O•
]

# --- Alkenes ---
ALLYLIC_CATIONS = [
    Formula({"C": 3, "H": 5}),          # C3H5+
]


# ------------------------------------------------------------
# Helper: subtract a loss from a parent formula
# ------------------------------------------------------------
def _subtract(parent: Formula, loss: Formula) -> Formula | None:
    """
    Subtract a loss formula from the parent formula.
    Returns None if subtraction is impossible.
    """
    result: Dict[str, int] = {}

    for el, cnt in parent.elements.items():
        new = cnt - loss.elements.get(el, 0)
        if new < 0:
            return None
        if new > 0:
            result[el] = new

    for el in loss.elements:
        if el not in parent.elements:
            return None

    if not result:
        return None

    return Formula(result)


# ------------------------------------------------------------
# Helper: check if cation is subset of parent
# ------------------------------------------------------------
def _subset_of_parent(cat: Formula, parent: Formula) -> bool:
    for el, cnt in cat.elements.items():
        if el not in parent.elements:
            return False
        if cnt > parent.elements[el]:
            return False
    return True


# ------------------------------------------------------------
# Functional group detection (robust to missing elements)
# ------------------------------------------------------------
def detect_functional_groups(parent: Formula) -> Dict[str, bool]:
    elems = parent.elements
    fg: Dict[str, bool] = {
        "alcohol": False,
        "carbonyl": False,
        "aromatic": False,
        "amine": False,
        "ester": False,
        "ether": False,
        "alkene": False,
        "halogen": False,
    }

    c = elems.get("C", 0)
    h = elems.get("H", 0)
    o = elems.get("O", 0)
    n = elems.get("N", 0)
    cl = elems.get("Cl", 0)
    br = elems.get("Br", 0)

    # Very simple heuristics; you likely already had something similar
    if o > 0 and h >= 2:
        fg["alcohol"] = True

    if o > 0 and c >= 1:
        fg["carbonyl"] = True  # generic carbonyl-like

    if dbe(parent) >= 4 and c >= 6:
        fg["aromatic"] = True

    if n > 0:
        fg["amine"] = True

    if o >= 2 and c >= 2:
        fg["ester"] = True

    if o > 0 and c >= 2:
        fg["ether"] = True

    if dbe(parent) >= 1 and c >= 2:
        fg["alkene"] = True

    if cl > 0 or br > 0:
        fg["halogen"] = True

    return fg


# ------------------------------------------------------------
# Main rule-based fragment generator
# ------------------------------------------------------------
def generate_rule_based_fragments(
    parent: Formula,
    rule_flags: Dict[str, bool],
    auto_detect_rules: bool = True,
) -> List[Tuple[Formula, str]]:
    """
    Generate fragments from a parent formula using rule families.

    Returns a list of (fragment_formula, rule_source).
    """
    results: List[Tuple[Formula, str]] = []
    elems = parent.elements

    # Functional group detection
    if auto_detect_rules:
        fg = detect_functional_groups(parent)
    else:
        fg = {k: True for k in [
            "alcohol", "carbonyl", "aromatic", "amine",
            "ester", "ether", "alkene", "halogen"
        ]}

    # --------------------------------------------------------
    # Universal neutral losses
    # --------------------------------------------------------
    if rule_flags.get("neutral_losses", True):
        for loss in NEUTRAL_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"neutral_loss_{loss.to_string()}"))

    # --------------------------------------------------------
    # Double neutral losses (optional)
    # --------------------------------------------------------
    if rule_flags.get("double_neutral_losses", False):
        for loss1 in NEUTRAL_LOSSES:
            frag1 = _subtract(parent, loss1)
            if frag1 is None:
                continue
            for loss2 in NEUTRAL_LOSSES:
                frag2 = _subtract(frag1, loss2)
                if frag2 is not None:
                    results.append((frag2, f"double_neutral_loss_{loss1.to_string()}_{loss2.to_string()}"))

    # --------------------------------------------------------
    # Common cations
    # --------------------------------------------------------
    if rule_flags.get("common_cations", True):
        for cat in COMMON_CATIONS:
            if _subset_of_parent(cat, parent):
                results.append((cat, "common_cation"))

    # --------------------------------------------------------
    # Alpha-cleavage (generic)
    # --------------------------------------------------------
    if rule_flags.get("alpha_cleavage", True):
        for loss in ALPHA_CLEAVAGE_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"alpha_cleavage_{loss.to_string()}"))

    # --------------------------------------------------------
    # Rearrangement losses
    # --------------------------------------------------------
    if rule_flags.get("rearrangements", True):
        for loss in REARRANGEMENT_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"rearrangement_{loss.to_string()}"))

    # --------------------------------------------------------
    # Oxygen-adjacent losses
    # --------------------------------------------------------
    if rule_flags.get("oxygen_adjacent", True) and elems.get("O", 0) > 0:
        for loss in OXYGEN_ADJACENT_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"oxygen_adjacent_{loss.to_string()}"))

    # --------------------------------------------------------
    # Hydrogen transfer
    # --------------------------------------------------------
    if rule_flags.get("hydrogen_transfer", True):
        for loss in HYDROGEN_TRANSFER_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"hydrogen_transfer_{loss.to_string()}"))

    # --------------------------------------------------------
    # McLafferty rearrangements
    # --------------------------------------------------------
    if rule_flags.get("mclafferty", True) and fg.get("carbonyl", False):
        for loss in MCLAFFERTY_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"mclafferty_{loss.to_string()}"))

    # ========================================================
    # Functional-group–specific rules
    # ========================================================

    # --- Alcohols ---
    if rule_flags.get("alcohol_rules", True) and fg.get("alcohol", False):
        for loss in ALCOHOL_BETA_CLEAVAGE_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"alcohol_beta_cleavage_{loss.to_string()}"))

        for loss in ALCOHOL_DEHYDRATION_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"alcohol_dehydration_{loss.to_string()}"))

        for loss in ALCOHOL_GAMMA_H_SHIFT_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"alcohol_gamma_H_shift_{loss.to_string()}"))

    # --- Carbonyls ---
    if rule_flags.get("carbonyl_rules", True) and fg.get("carbonyl", False):
        for ion in ACYLIUM_IONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "acylium_ion"))

        for loss in CARBONYL_ALPHA_CLEAVAGE_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"carbonyl_alpha_cleavage_{loss.to_string()}"))

    # --- Aromatics ---
    if rule_flags.get("aromatic_rules", True) and fg.get("aromatic", False):
        for ion in AROMATIC_TROPYLIUM_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "tropylium_cation"))

        for ion in AROMATIC_PHENYL_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "phenyl_cation"))

        for ion in AROMATIC_BENZYL_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "benzyl_cation"))

        for loss in AROMATIC_RING_CONTRACTION_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"aromatic_ring_contraction_{loss.to_string()}"))

    # --- Amines ---
    if rule_flags.get("amine_rules", True) and fg.get("amine", False):
        for ion in IMINIUM_IONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "iminium_ion"))

        for loss in AMINE_ALPHA_CLEAVAGE_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"amine_alpha_cleavage_{loss.to_string()}"))

    # --- Esters ---
    if rule_flags.get("ester_rules", True) and fg.get("ester", False):
        for ion in ESTER_ACYLIUM_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "ester_acylium_cation"))

        for loss in ESTER_ALKOXY_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"ester_alkoxy_loss_{loss.to_string()}"))

    # --- Halogens ---
    if rule_flags.get("halogen_rules", True) and fg.get("halogen", False):
        for loss in HALOGEN_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"halogen_loss_{loss.to_string()}"))

        for ion in HALOGEN_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "halogen_cation"))

    # --- Ethers ---
    if rule_flags.get("ether_rules", True) and fg.get("ether", False):
        for loss in ETHER_ALPHA_CLEAVAGE_LOSSES:
            frag = _subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"ether_alpha_cleavage_{loss.to_string()}"))

    # --- Alkenes ---
    if rule_flags.get("alkene_rules", True) and fg.get("alkene", False):
        for ion in ALLYLIC_CATIONS:
            if _subset_of_parent(ion, parent):
                results.append((ion, "allylic_cation"))

    # --------------------------------------------------------
    # Oxygen-specific SECONDARY fragmentation rules
    # (work on any oxygen-containing fragment, not just parent)
    # --------------------------------------------------------
    o = elems.get("O", 0)
    c = elems.get("C", 0)
    h = elems.get("H", 0)

    if o >= 1 and c >= 2:
        # 1. Carbonyl secondary cleavage: R-CO-R' → RCO+ + R'•
        frag2 = _subtract(parent, Formula({"C": 1, "O": 1}))
        if frag2 is not None:
            results.append((frag2, "carbonyl_secondary"))

    if o >= 1 and c >= 1:
        # 2. Formyl cation formation: R-CO-R' → CHO+
        cho = Formula({"C": 1, "H": 1, "O": 1})
        if _subset_of_parent(cho, parent):
            results.append((cho, "formyl_cation"))

    if o >= 1 and c >= 2:
        # 3. Acyl cation formation: R-CO-R' → CO+ (generic acyl core)
        co = Formula({"C": 1, "O": 1})
        if _subset_of_parent(co, parent):
            results.append((co, "acyl_cation"))

    if o >= 1 and c >= 1:
        # 4. Alkoxy cation formation: R-O-R' → CO+ (generic RO+ surrogate)
        co = Formula({"C": 1, "O": 1})
        if _subset_of_parent(co, parent):
            results.append((co, "alkoxy_cation"))

    if o >= 1 and c >= 1:
        # 5. Secondary CO loss: RCO+ → R+ + CO
        frag2 = _subtract(parent, Formula({"C": 1, "O": 1}))
        if frag2 is not None:
            results.append((frag2, "CO_loss_secondary"))

    if o >= 1 and c >= 1 and h >= 2:
        # 6. Loss of CH2O (formaldehyde): R-CH2-OH → R+ + CH2O
        frag2 = _subtract(parent, Formula({"C": 1, "H": 2, "O": 1}))
        if frag2 is not None:
            results.append((frag2, "CH2O_loss"))

    return results