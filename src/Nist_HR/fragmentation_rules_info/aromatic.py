# fragmentation_rules/aromatic.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula
from src.Nist_HR.fragmentation_rules_info.base import subtract, subset_of_parent


# ------------------------------------------------------------
# Core aromatic ions
# ------------------------------------------------------------
TROPYLIUM = Formula({"C": 7, "H": 7})
PHENYL    = Formula({"C": 6, "H": 5})
BENZYL    = Formula({"C": 7, "H": 7})   # same formula as tropylium

# Ring contraction series
RING_CONTRACTION_LOSSES = [
    Formula({"C": 2, "H": 2}),   # C2H2
    Formula({"C": 1, "H": 2}),   # CH2
]

# Carbonyl-specific aromatic ions
AR_CO_IONS = [
    Formula({"C": 7, "H": 5, "O": 1}),   # benzoyl cation C7H5O+
    Formula({"C": 6, "H": 5}),           # phenyl cation
]


# ------------------------------------------------------------
# Aromatic rule generator
# ------------------------------------------------------------
def generate(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    results: List[Tuple[Formula, str]] = []

    if not fg.get("aromatic", False):
        return results

    # --------------------------------------------------------
    # 1. Tropylium cation
    # --------------------------------------------------------
    if subset_of_parent(TROPYLIUM, parent):
        results.append((TROPYLIUM, "aromatic_tropylium"))

    # --------------------------------------------------------
    # 2. Phenyl cation
    # --------------------------------------------------------
    if subset_of_parent(PHENYL, parent):
        results.append((PHENYL, "aromatic_phenyl"))

    # --------------------------------------------------------
    # 3. Benzyl cation (benzylic cleavage)
    # --------------------------------------------------------
    if fg.get("benzylic", False):
        if subset_of_parent(BENZYL, parent):
            results.append((BENZYL, "aromatic_benzylic_cleavage"))

    # --------------------------------------------------------
    # 4. Side-chain loss (Ar–R → Ar+)
    # --------------------------------------------------------
    if fg.get("side_chain", False):
        # subtract side chain formula if available
        side = fg.get("side_chain_formula")
        if side:
            frag = subtract(parent, side)
            if frag:
                results.append((frag, "aromatic_side_chain_loss"))

    # --------------------------------------------------------
    # 5. Carbonyl-specific aromatic ions
    # --------------------------------------------------------
    if fg.get("aromatic_carbonyl", False):
        for ion in AR_CO_IONS:
            if subset_of_parent(ion, parent):
                results.append((ion, "aromatic_carbonyl_fragment"))

    # --------------------------------------------------------
    # 6. Ortho-cleavage (heteroatom-assisted)
    # --------------------------------------------------------
    if fg.get("phenol", False) or fg.get("aniline", False) or fg.get("anisole", False):
        # C6H4X+ type ions
        base = Formula({"C": 6, "H": 4})
        if subset_of_parent(base, parent):
            results.append((base, "aromatic_ortho_cleavage"))

    # --------------------------------------------------------
    # 7. Ring contraction series
    # --------------------------------------------------------
    for loss in RING_CONTRACTION_LOSSES:
        frag = subtract(parent, loss)
        if frag:
            results.append((frag, f"aromatic_ring_contraction_{loss.to_string()}"))

    return results