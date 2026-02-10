"""
chemistry.py

Chemical utility functions used across the fragment enumeration
and assignment pipeline.

Uses explicit monoisotopic masses (IUPAC) instead of RDKit's
average atomic weights.
"""

from __future__ import annotations

from formula import Formula

# Monoisotopic atomic masses (Da)
MONO_MASS = {
    "H": 1.00782503223,
    "D": 2.0141017778,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
    "P": 30.9737619985,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "F": 18.998403163,
    "I": 126.904473,
}

ELECTRON_MASS_DA = 0.000548579909065


def exact_mass(formula: Formula, charged: bool = True) -> float:
    """
    Compute the exact monoisotopic mass of a formula.

    Parameters
    ----------
    formula : Formula
        Molecular formula with element counts.
    charged : bool
        If True, subtract one electron mass (EI radical cation).

    Returns
    -------
    float
        Exact monoisotopic mass.
    """
    mass = 0.0
    for el, count in formula.elements.items():
        try:
            mass += MONO_MASS[el] * count
        except KeyError:
            raise ValueError(f"Unknown element in formula: {el}")

    if charged:
        mass -= ELECTRON_MASS_DA

    return mass


def dbe(formula: Formula) -> float:
    """
    Compute double bond equivalents (RDBE).

    DBE = C - H/2 + N/2 + 1 - X/2
    where X = halogens (F, Cl, Br, I).
    """
    elems = formula.elements

    C = elems.get("C", 0)
    H = elems.get("H", 0)
    N = elems.get("N", 0)
    X = elems.get("F", 0) + elems.get("Cl", 0) + elems.get("Br", 0) + elems.get("I", 0)

    return C - H / 2.0 + N / 2.0 + 1.0 - X / 2.0