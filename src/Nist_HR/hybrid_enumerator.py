# hybrid_enumerator.py

from __future__ import annotations
from typing import Dict, List, Tuple

from formula import Formula
from chemistry import exact_mass

from peak_driven_enumerator import PeakDrivenEnumerator
from MLFF_fragmentation.bde_enumerator import BDEDrivenEnumerator


class HybridEnumerator:
    """
    Combines:
        - Rule-based formula fragmentation
        - BDE-driven structural fragmentation

    Produces a unified set of:
        (Formula, rule_source, depth)
    indexed by nominal mass.
    """

    def __init__(
        self,
        parent_formula: Formula,
        mol,
        bond_data,
        rule_flags: Dict[str, bool],
        max_depth: int = 2,
        bde_threshold: float = 100.0,
        bde_softness: float = 25.0,
    ):
        self.parent_formula = parent_formula
        self.max_depth = max_depth

        # -----------------------------
        # 1. Rule-based enumerator
        # -----------------------------
        self.rule_enum = PeakDrivenEnumerator(
            parent_formula,
            rule_flags=rule_flags,
            max_depth=max_depth,
            auto_detect_rules=True,
        )

        # -----------------------------
        # 2. BDE-driven enumerator
        # -----------------------------
        self.bde_enum = BDEDrivenEnumerator(
            parent_formula,
            mol,
            bond_data,
            max_depth=max_depth,
            threshold=bde_threshold,
            softness=bde_softness,
        )

        # -----------------------------
        # 3. Merge fragments
        # -----------------------------
        self.lookup = self._merge_enumerators()

    # ------------------------------------------------------------
    # Merge rule-based + BDE-based fragments
    # ------------------------------------------------------------
    def _merge_enumerators(self):
        merged = {}

        # Helper to insert fragments
        def add_frag(frag: Formula, rule: str, depth: int):
            key = frag.to_string()
            if key not in merged:
                merged[key] = {
                    "frag": frag,
                    "rule_sources": {rule},
                    "depth": depth,
                }
            else:
                merged[key]["rule_sources"].add(rule)
                merged[key]["depth"] = min(merged[key]["depth"], depth)

        # Add rule-based fragments
        for nm, frags in self.rule_enum.lookup.items():
            for frag, rule, depth in frags:
                add_frag(frag, rule, depth)

        # Add BDE-based fragments
        for nm, frags in self.bde_enum.lookup.items():
            for frag, rule, depth in frags:
                add_frag(frag, rule, depth)

        # Build final lookup by nominal mass
        lookup = {}
        for key, info in merged.items():
            frag = info["frag"]
            rule_source = " + ".join(sorted(info["rule_sources"]))
            depth = info["depth"]

            nm = round(exact_mass(frag))
            lookup.setdefault(nm, []).append((frag, rule_source, depth))

        return lookup

    # ------------------------------------------------------------
    # API required by PeakDrivenAssignmentEngine
    # ------------------------------------------------------------
    def enumerate_for_peak(self, target_nominal_mz: int):
        return self.lookup.get(target_nominal_mz, [])