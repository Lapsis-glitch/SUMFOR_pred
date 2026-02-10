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
    etc.
    """

    def __init__(self, parent: Formula, rule_flags: Dict[str, bool], max_depth: int = 1):
        self.parent = parent
        self.rule_flags = rule_flags
        self.max_depth = max_depth

    def generate(self) -> List[Tuple[Formula, str, int]]:
        """
        Returns a list of (fragment_formula, rule_source, depth)
        where depth = 1 (primary), 2 (secondary), etc.
        """
        results: Dict[str, Tuple[Formula, str, int]] = {}
        visited: Set[str] = set()

        # Depth 1: primary fragments
        primary = generate_rule_based_fragments(self.parent, self.rule_flags)
        for frag, rule in primary:
            key = frag.to_string()
            results[key] = (frag, rule, 1)
            visited.add(key)

        current_level = primary

        # Depth >= 2: recursive fragmentation
        for depth in range(2, self.max_depth + 1):
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