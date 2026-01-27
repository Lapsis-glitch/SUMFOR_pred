"""
enumerator.py

Defines FragmentEnumerator:
- Applies rule-based fragment generation
- Converts (Formula, rule_source) into Fragment objects
- Computes exact mass, nominal mass, DBE
- Applies NIST binning

This module contains no chemistry logic.
"""

from __future__ import annotations
from typing import List, Dict

from formula import Formula
from fragment import Fragment
from chemistry import exact_mass, dbe
from fragmentation_rules import generate_rule_based_fragments
from utils import nist_bin


class FragmentEnumerator:
    """
    Generates Fragment objects from a parent Formula using rule-based
    fragmentation logic.
    """

    def __init__(self, parent: Formula):
        self.parent = parent
        self._fragments: List[Fragment] = []

    def enumerate_subformulas(self) -> List[Fragment]:
        """
        Generate all fragments using universal EI rules.

        Returns
        -------
        list of Fragment
        """

        # Universal rule set (all enabled by default)
        rule_flags = {
            "neutral_losses": True,
            "double_neutral_losses": True,
            "common_cations": True,
            "restrict_cations_to_parent_elements": True,
            "alpha_cleavage": True,
            "rearrangements": True,
            "oxygen_adjacent": True,
            "hydrogen_transfer": True,
            "mclafferty": True,
        }

        # Returns list of (Formula, rule_source)
        formulas = generate_rule_based_fragments(self.parent, rule_flags)

        fragments: List[Fragment] = []

        for f, rule_source in formulas:
            mass = exact_mass(f)
            nom = nist_bin(mass)   # NIST asymmetric binning
            if nom is None:
                continue

            frag = Fragment(
                formula=f,
                exact_mass=mass,
                nominal_mass=nom,
                dbe=dbe(f),
                rule_source=rule_source,
            )
            fragments.append(frag)

        self._fragments = fragments
        return fragments

    @property
    def fragments(self) -> List[Fragment]:
        """
        Cached access to generated fragments.
        """
        if not self._fragments:
            self.enumerate_subformulas()
        return self._fragments

    def build_nominal_lookup(self) -> Dict[int, List[Fragment]]:
        """
        Build dictionary mapping nominal mass → list of fragments.
        """
        lookup: Dict[int, List[Fragment]] = {}
        for frag in self.fragments:
            lookup.setdefault(frag.nominal_mass, []).append(frag)
        return lookup