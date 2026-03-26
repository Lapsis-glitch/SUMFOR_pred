# recursive_fragmenter.py

from __future__ import annotations
from typing import Dict, List, Tuple, Set
from formula import Formula
from fragmentation_rules import generate_rule_based_fragments


class RecursiveFragmenter:
    """
    Wraps the existing rule-based fragment generator and applies it recursively.

    max_depth = 1 → primary fragments only
    max_depth = 2 → secondary fragments
    max_depth = 3 → tertiary fragments
    """

    def __init__(self, parent: Formula, rule_flags: Dict[str, bool], max_depth: int = 1):
        self.parent = parent
        self.rule_flags = rule_flags
        self.max_depth = max_depth

    def generate(self, start_depth: int = 1) -> List[Tuple[Formula, str, int]]:
        """
        Returns a list of (fragment_formula, rule_source, depth)
        where depth = start_depth (primary), start_depth+1 (secondary), etc.
        """
        results: Dict[str, Tuple[Formula, str, int]] = {}
        visited: Set[str] = set()

        # Depth = start_depth: primary fragments
        primary = generate_rule_based_fragments(self.parent, self.rule_flags)
        for frag, rule in primary:
            key = frag.to_string()
            results[key] = (frag, rule, start_depth)
            visited.add(key)

        current_level = primary

        # Depth >= start_depth+1: recursive fragmentation
        for depth in range(start_depth + 1, self.max_depth + 1):
            next_level = []

            for frag, rule in current_level:
                subfrags = generate_rule_based_fragments(frag, self.rule_flags)

                for f2, rule2 in subfrags:
                    key = f2.to_string()
                    if key in visited:
                        continue

                    visited.add(key)
                    results[key] = (f2, rule2, depth)
                    next_level.append((f2, rule2))

            current_level = next_level

            if not current_level:
                break

        return list(results.values())

    # NEW: public API for HybridEnumerator
    def fragment_formula(self, formula: Formula, start_depth: int = 1):
        fragger = RecursiveFragmenter(formula, self.rule_flags, max_depth=self.max_depth)
        return fragger.generate(start_depth=start_depth)