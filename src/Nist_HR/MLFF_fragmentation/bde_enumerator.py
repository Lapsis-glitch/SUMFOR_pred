# bde_enumerator.py

from __future__ import annotations
from typing import Dict, List, Tuple

from formula import Formula
from chemistry import exact_mass
from .bde_fragmenter import recursive_fragment   # your Mol-based BDE tree builder


class BDEDrivenEnumerator:
    """
    Drop-in replacement for PeakDrivenEnumerator, but using
    BDE-driven structural fragmentation instead of rule-based
    formula fragmentation.

    Produces:
        (Formula, rule_source, depth)
    and indexes them by nominal mass.
    """

    def __init__(
        self,
        parent_formula: Formula,
        mol,
        bond_data,
        max_depth: int = 5,
        threshold: float = 120.0,
        softness: float = 25.0,
    ):
        self.parent_formula = parent_formula
        self.mol = mol
        self.bond_data = bond_data
        self.max_depth = max_depth
        self.threshold = threshold
        self.softness = softness

        # 1) Build the Mol-based fragmentation tree
        self.tree = recursive_fragment(
            mol,
            bond_data,
            max_depth=max_depth,
            threshold=threshold,
            softness=softness,
        )
        # print("\n=== DEBUG: ROOT NODE FROM BDE FRAGMENTER ===")
        # print(self.tree)
        # print("=== END DEBUG ===\n")

        # 2) Flatten into (Formula, rule_source, depth)
        flat = self._flatten_tree(self.tree)
        self.debug_total_bde_frags = len(flat)

        # 3) Build lookup by nominal mass
        self.lookup = {}
        for frag_formula, rule_source, depth in flat:
            nm = round(exact_mass(frag_formula))
            self.lookup.setdefault(nm, []).append(
                (frag_formula, rule_source, depth)
            )
        # self.debug_surviving_frags = sum(len(v) for v in self.lookup.values())
        #
        # print(f"[BDE DEBUG] Entry: {self.parent_formula}")
        # print(f"  Total BDE fragments generated: {self.debug_total_bde_frags}")
        # print(f"  Surviving after physics rules: {self.debug_surviving_frags}")
        # print(f"  Eliminated: {self.debug_total_bde_frags - self.debug_surviving_frags}")
        # print()

    # ------------------------------------------------------------
    # Flatten the BDE fragmentation tree
    # ------------------------------------------------------------
    def _flatten_tree(self, node, depth=1):
        """
        Convert the Mol-based tree into:
            (Formula, rule_source, depth)
        """
        out = []

        # Convert formula string → Formula object
        # frag_formula = Formula.from_string(node["formula"])
        raw = node["formula"]

        # raw is now a Formula object
        if isinstance(raw, Formula):
            frag_formula = raw
        else:
            # fallback for safety
            frag_formula = Formula.from_string(raw)

        rule_source = f"bde_depth_{depth}"

        out.append((frag_formula, rule_source, depth))

        for child in node.get("children", []):
            out.extend(self._flatten_tree(child, depth + 1))

        return out

    # ------------------------------------------------------------
    # API required by PeakDrivenAssignmentEngine
    # ------------------------------------------------------------
    def enumerate_for_peak(self, target_nominal_mz: int):
        """
        Return all fragments whose nominal mass matches the peak.
        """
        return self.lookup.get(target_nominal_mz, [])