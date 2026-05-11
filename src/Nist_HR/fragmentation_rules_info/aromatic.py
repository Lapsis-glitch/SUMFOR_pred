# fragmentation_rules/aromatic.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from .base import subtract, subset_of_parent


# ------------------------------------------------------------
# Core aromatic ions
# ------------------------------------------------------------
TROPYLIUM = Formula({"C": 7, "H": 7})
PHENYL    = Formula({"C": 6, "H": 5})
BENZYL    = Formula({"C": 7, "H": 7})   # same formula as tropylium

# C4: Cyclopentadienyl cation (m/z 65) — very common ring contraction product
CYCLOPENTADIENYL = Formula({"C": 5, "H": 5})

# Ring contraction series
RING_CONTRACTION_LOSSES = [
    Formula({"C": 2, "H": 2}),   # C2H2
    Formula({"C": 1, "H": 2}),   # CH2
]

# C3: Retro-Diels-Alder losses
# C2H2 and C2H4 already in universal neutral losses; add the larger ones here
RDA_LOSSES = [
    Formula({"C": 3, "H": 4}),   # C3H4 (40 Da) — allene / propyne
    Formula({"C": 4, "H": 6}),   # C4H6 (54 Da) — butadiene
]

# Carbonyl-specific aromatic ions
AR_CO_IONS = [
    Formula({"C": 7, "H": 5, "O": 1}),   # benzoyl cation C7H5O+
    Formula({"C": 6, "H": 5}),           # phenyl cation
]

# C7: CO loss — phenol → cyclopentadiene (C5H6, m/z 66)
CO_LOSS = Formula({"C": 1, "O": 1})


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
    # 3. C4: Cyclopentadienyl cation (m/z 65)
    # --------------------------------------------------------
    if subset_of_parent(CYCLOPENTADIENYL, parent):
        results.append((CYCLOPENTADIENYL, "aromatic_cyclopentadienyl"))

    # --------------------------------------------------------
    # 4. Benzyl cation (benzylic cleavage)
    # --------------------------------------------------------
    if fg.get("benzylic", False):
        if subset_of_parent(BENZYL, parent):
            results.append((BENZYL, "aromatic_benzylic_cleavage"))

    # --------------------------------------------------------
    # 5. Side-chain loss (Ar–R → Ar+)
    # --------------------------------------------------------
    if fg.get("side_chain", False):
        # subtract side chain formula if available
        side = fg.get("side_chain_formula")
        if side:
            frag = subtract(parent, side)
            if frag:
                results.append((frag, "aromatic_side_chain_loss"))

    # --------------------------------------------------------
    # 6. Carbonyl-specific aromatic ions
    # --------------------------------------------------------
    if fg.get("aromatic_carbonyl", False):
        for ion in AR_CO_IONS:
            if subset_of_parent(ion, parent):
                results.append((ion, "aromatic_carbonyl_fragment"))

    # --------------------------------------------------------
    # 7. Ortho-cleavage (heteroatom-assisted)
    # --------------------------------------------------------
    if fg.get("phenol", False) or fg.get("aniline", False) or fg.get("anisole", False):
        # C6H4X+ type ions
        base = Formula({"C": 6, "H": 4})
        if subset_of_parent(base, parent):
            results.append((base, "aromatic_ortho_cleavage"))

    # --------------------------------------------------------
    # 8. Ring contraction series
    # --------------------------------------------------------
    for loss in RING_CONTRACTION_LOSSES:
        frag = subtract(parent, loss)
        if frag:
            results.append((frag, f"aromatic_ring_contraction_{loss.to_string()}"))

    # --------------------------------------------------------
    # 9. C7: CO loss from aromatic compounds (phenol → C5H6)
    #    Given a higher-priority label for FPS prior
    # --------------------------------------------------------
    if parent.elements.get("O", 0) > 0:
        frag = subtract(parent, CO_LOSS)
        if frag:
            results.append((frag, "aromatic_CO_loss"))

    # --------------------------------------------------------
    # 10. C3: Retro-Diels-Alder losses (C3H4, C4H6)
    # --------------------------------------------------------
    for loss in RDA_LOSSES:
        frag = subtract(parent, loss)
        if frag:
            results.append((frag, f"rda_loss_{loss.to_string()}"))

    return results