from __future__ import annotations
from typing import Dict
from src.Nist_HR.formula import Formula
from chemistry import dbe


# ------------------------------------------------------------
# Helper: subtract a loss from a parent formula
# ------------------------------------------------------------
def subtract(parent: Formula, loss: Formula) -> Formula | None:
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

    # Ensure no new elements appear
    for el in loss.elements:
        if el not in parent.elements:
            return None

    if not result:
        return None

    return Formula(result)


# ------------------------------------------------------------
# Helper: check if cation is subset of parent
# ------------------------------------------------------------
def subset_of_parent(cat: Formula, parent: Formula) -> bool:
    for el, cnt in cat.elements.items():
        if el not in parent.elements:
            return False
        if cnt > parent.elements[el]:
            return False
    return True


# ------------------------------------------------------------
# Functional group detection
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

        # sulfur families
        "sulfur": False,
        "thiol_or_thioether": False,
        "sulfonyl": False,

        # phosphorus
        "phosphorus": False,
    }

    c = elems.get("C", 0)
    h = elems.get("H", 0)
    o = elems.get("O", 0)
    n = elems.get("N", 0)
    s = elems.get("S", 0)
    p = elems.get("P", 0)

    # --- sulfur ---
    fg["sulfur"] = s > 0
    fg["thiol_or_thioether"] = (s > 0 and o == 0)
    fg["sulfonyl"] = (s > 0 and o >= 2)

    # --- phosphorus ---
    fg["phosphorus"] = p > 0

    # --- oxygen families ---
    if o > 0 and h >= 2:
        fg["alcohol"] = True

    if o > 0 and c >= 1:
        fg["carbonyl"] = True

    # --- aromaticity ---
    if dbe(parent) >= 4 and c >= 6:
        fg["aromatic"] = True

    # --- nitrogen ---
    if n > 0:
        fg["amine"] = True

    # --- esters ---
    if o >= 2 and c >= 2:
        fg["ester"] = True

    # --- ethers ---
    if o > 0 and c >= 2:
        fg["ether"] = True

    # --- alkenes ---
    if dbe(parent) >= 1 and c >= 2:
        fg["alkene"] = True

    # --- halogens ---
    f = elems.get("F", 0)
    cl = elems.get("Cl", 0)
    br = elems.get("Br", 0)
    i = elems.get("I", 0)

    if f > 0 or cl > 0 or br > 0 or i > 0:
        fg["halogen"] = True

    return fg