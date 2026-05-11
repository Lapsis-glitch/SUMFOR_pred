# fragmentation_rules/amine.py

from __future__ import annotations
from typing import List, Tuple
from formula import Formula
from chemistry import dbe
from .base import subtract,subset_of_parent

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

# C1: HCN / HNC loss — diagnostic for N-heterocycles
HCN_LOSS = Formula({"H": 1, "C": 1, "N": 1})   # HCN (27 Da)

# C8: CN radical loss — diagnostic for nitriles
CN_LOSS = Formula({"C": 1, "N": 1})             # CN• (26 Da)

# C2: NO and NO2 losses — dominant for nitro compounds
NO_LOSS  = Formula({"N": 1, "O": 1})            # NO• (30 Da)
NO2_LOSS = Formula({"N": 1, "O": 2})            # NO2• (46 Da)

# C6: Amide-specific losses
AMIDE_LOSSES = [
    Formula({"C": 1, "H": 1, "N": 1, "O": 1}),  # CHNO  (43 Da) — isocyanate loss
    Formula({"C": 1, "H": 2, "N": 1, "O": 1}),  # CH2NO (44 Da) — formamide loss
    Formula({"C": 1, "H": 3, "N": 1}),            # CH3N  (29 Da) — methylamine loss
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
    Generate amine-specific EI fragments, including:
    - iminium ions
    - alpha cleavage next to nitrogen
    - beta-N losses
    - HCN loss (N-heterocycles)
    - CN radical loss (nitriles)
    - NO / NO2 losses (nitro compounds)
    - amide losses (when carbonyl also present)
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

    # --- Beta-N losses ---
    results.extend(generate_beta_n(parent, fg))

    # --- C1: HCN loss (27 Da) — N-heterocycles ---
    # Gate: amine + high DBE (suggesting N-heterocycle)
    if dbe(parent) >= 4:
        frag = subtract(parent, HCN_LOSS)
        if frag is not None:
            results.append((frag, "amine_HCN_loss"))

    # --- C8: CN radical loss (26 Da) — nitriles ---
    # Gate: amine + DBE >= 2 (C≡N contributes 2)
    if dbe(parent) >= 2:
        frag = subtract(parent, CN_LOSS)
        if frag is not None:
            results.append((frag, "nitrile_CN_loss"))

    # --- C2: NO loss (30 Da) — nitro compounds, N-oxides ---
    elems = parent.elements
    if elems.get("O", 0) >= 1:
        frag = subtract(parent, NO_LOSS)
        if frag is not None:
            results.append((frag, "nitro_NO_loss"))

    # --- C2: NO2 loss (46 Da) — nitro compounds ---
    if elems.get("O", 0) >= 2:
        frag = subtract(parent, NO2_LOSS)
        if frag is not None:
            results.append((frag, "nitro_NO2_loss"))

    # --- C6: Amide losses — when amine + carbonyl are both present ---
    if fg.get("carbonyl", False):
        for loss in AMIDE_LOSSES:
            frag = subtract(parent, loss)
            if frag is not None:
                results.append((frag, f"amide_loss_{loss.to_string()}"))

    return results