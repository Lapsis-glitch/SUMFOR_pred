# peak_driven_enumerator.py

from __future__ import annotations
from typing import Dict, List, Tuple
from formula import Formula
from chemistry import exact_mass
from recursive_fragmenter import RecursiveFragmenter


class PeakDrivenEnumerator:
    """
    Enumerates only subformulas that:
    - are element-wise <= parent
    - match a target nominal m/z
    - satisfy DBE >= 0
    - are explainable by at least one fragmentation rule
    - optionally include secondary/tertiary/... fragments
    """

    def __init__(self, parent: Formula, rule_flags: Dict[str, bool],
                 max_depth: int = 1, auto_detect_rules=True):

        self.parent = parent
        self.rule_flags = rule_flags
        self.max_depth = max_depth
        self.auto_detect_rules = auto_detect_rules

        # Generate recursive fragments
        fraggen = RecursiveFragmenter(parent, rule_flags, max_depth=max_depth)
        recursive_frags = fraggen.generate()   # (frag, rule, depth)

        # Build lookup by nominal mass
        self.lookup = {}
        for frag, rule, depth in recursive_frags:
            nm = round(exact_mass(frag))
            self.lookup.setdefault(nm, []).append((frag, rule, depth))

    def enumerate_for_peak(self, target_nominal_mz: int) -> List[Tuple[Formula, str, int]]:
        """
        Return only fragments that match the target nominal m/z.
        """
        return self.lookup.get(target_nominal_mz, [])