"""
fragment.py

Defines the Fragment dataclass: a lightweight container representing a
single fragment formula produced during enumeration.

A Fragment stores:
- the fragment's molecular formula
- its exact monoisotopic mass
- its nominal (integer) mass
- its DBE (double bond equivalents)
- its rule_source (fragmentation rule family)

This module contains no chemistry logic and no enumeration logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from formula import Formula


@dataclass
class Fragment:
    """
    Represents a chemically plausible fragment formula.

    Attributes
    ----------
    formula : Formula
        The fragment's molecular formula.
    exact_mass : float
        Exact monoisotopic mass computed from element counts.
    nominal_mass : int
        Rounded integer mass (m/z) used for matching NIST peaks.
    dbe : float
        Double bond equivalents (RDBE), used for plausibility scoring.
    rule_source : str
        Fragmentation rule family that produced this fragment.
    """

    formula: Formula
    exact_mass: float
    nominal_mass: int
    dbe: float
    rule_source: str = "unknown"

    def to_dict(self) -> dict:
        """
        Convert the fragment into a serializable dictionary.
        """
        return {
            "formula": self.formula.to_string(),
            "exact_mass": self.exact_mass,
            "nominal_mass": self.nominal_mass,
            "dbe": self.dbe,
            "rule_source": self.rule_source,
        }

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """
        return (
            f"Fragment(formula={self.formula.to_string()}, "
            f"exact_mass={self.exact_mass:.5f}, "
            f"nominal_mass={self.nominal_mass}, "
            f"dbe={self.dbe:.2f}, "
            f"rule_source={self.rule_source})"
        )