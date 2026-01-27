"""
peakassignment.py

Defines:
- PeakAssignment: a lightweight container for assignment results

This version supports multi-match mode:
each viable fragment for a given m/z is emitted as its own PeakAssignment.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PeakAssignment:
    """
    Represents the assignment of a single NIST peak to one fragment candidate.

    Attributes
    ----------
    nominal_mz : int
        Integer m/z from the NIST EI spectrum.
    intensity : float
        Peak intensity (relative or absolute).
    best_exact_mz : float or None
        Exact mass of this fragment candidate.
    best_formula : str or None
        Formula string of this fragment candidate.
    candidate_formulas : list of str
        For multi-match mode, this will contain exactly [best_formula].
        For single-match mode, also [best_formula].
    confidence : float
        Confidence score in [0, 1].
    rule_source : str or None
        Fragmentation rule family that produced this fragment.
    """

    nominal_mz: int
    intensity: float
    best_exact_mz: Optional[float]
    best_formula: Optional[str]
    candidate_formulas: List[str]
    confidence: float
    rule_source: Optional[str] = None