"""
formula.py

Defines the Formula class: a lightweight, immutable molecular formula
representation used throughout the fragment enumeration and assignment
pipeline.

This module is intentionally minimal and dependency‑free.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Formula:
    """
    Represents a molecular formula as a mapping of element symbols to counts.

    Examples
    --------
    >>> Formula({"C": 6, "H": 6})
    >>> Formula.from_string("C6H6")
    """

    elements: Dict[str, int]

    def __post_init__(self):
        """
        Remove zero-count elements and enforce immutability.
        """
        cleaned = {el: n for el, n in self.elements.items() if n > 0}
        object.__setattr__(self, "elements", cleaned)

    @classmethod
    def from_string(cls, formula_str: str) -> Formula:
        """
        Parse a chemical formula string like 'C8H10O2' into a Formula object.

        Parameters
        ----------
        formula_str : str
            A string representation of a molecular formula.

        Returns
        -------
        Formula
        """
        import re
        tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula_str)
        elems: Dict[str, int] = {}

        for el, num in tokens:
            count = int(num) if num else 1
            elems[el] = elems.get(el, 0) + count

        return cls(elems)

    def to_string(self) -> str:
        """
        Convert the formula back into a canonical string representation.

        Returns
        -------
        str
            Example: {"C": 6, "H": 6} → "C6H6"
        """
        parts = []
        for el in sorted(self.elements.keys()):
            n = self.elements[el]
            parts.append(f"{el}{n}")
        return "".join(parts)

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """
        return f"Formula({self.to_string()})"