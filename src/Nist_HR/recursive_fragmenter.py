# recursive_fragmenter.py

from __future__ import annotations
from typing import Dict, List, Tuple, Set
from formula import Formula
from fragmentation_rules import generate_rule_based_fragments
from fragmentation_rules_info.base import is_radical_species, compute_loss


# D6: Even-electron rule toggle
# EE+ ions cannot lose radicals to form OE+• — this is chemically forbidden.
# Set to False to disable the filter (reverts to previous behaviour).
USE_EVEN_ELECTRON_RULE = True


class RecursiveFragmenter:
    """
    Wraps the existing rule-based fragment generator and applies it recursively.

    max_depth = 1 → primary fragments only
    max_depth = 2 → secondary fragments
    max_depth = 3 → tertiary fragments

    D6: Tracks electron state (OE+• vs EE+) through the fragmentation tree.
    The molecular ion is OE+• (odd-electron radical cation).
      - Losing a radical       → fragment becomes EE+ (even-electron)
      - Losing a neutral molecule → fragment stays OE+•
    The even-electron rule forbids EE+ → OE+• (i.e. EE+ cannot lose a radical).
    """

    def __init__(self, parent: Formula, rule_flags: Dict[str, bool],
                 max_depth: int = 1, parent_fg: Dict[str, bool] | None = None):
        self.parent = parent
        self.rule_flags = rule_flags
        self.max_depth = max_depth
        self.parent_fg = parent_fg   # D4: pre-computed FG dict (e.g. from SMARTS)

    def generate(self, start_depth: int = 1) -> List[Tuple[Formula, str, int]]:
        """
        Returns a list of (fragment_formula, rule_source, depth)
        where depth = start_depth (primary), start_depth+1 (secondary), etc.
        """
        results: Dict[str, Tuple[Formula, str, int]] = {}
        visited: Set[str] = set()

        # D6: Track electron state for each fragment
        # Key: formula string, Value: "OE" (odd-electron) or "EE" (even-electron)
        electron_state: Dict[str, str] = {}

        # ── Depth = start_depth: primary fragments from the molecular ion (OE+•)
        primary = generate_rule_based_fragments(
            self.parent, self.rule_flags, parent_fg=self.parent_fg
        )
        for frag, rule in primary:
            key = frag.to_string()

            # D6: Classify the loss to determine fragment electron state
            loss = compute_loss(self.parent, frag)
            if loss is not None:
                # Radical loss from OE+• → EE+;  neutral loss → OE+•
                electron_state[key] = "EE" if is_radical_species(loss) else "OE"
            else:
                # Direct cation (subset, not subtraction) → EE+ by convention
                electron_state[key] = "EE"

            results[key] = (frag, rule, start_depth)
            visited.add(key)

        current_level = primary

        # ── Depth >= start_depth+1: recursive fragmentation
        for depth in range(start_depth + 1, self.max_depth + 1):
            next_level = []

            for frag, rule in current_level:
                parent_key = frag.to_string()
                parent_state = electron_state.get(parent_key, "OE")

                subfrags = generate_rule_based_fragments(frag, self.rule_flags)

                for f2, rule2 in subfrags:
                    key = f2.to_string()
                    if key in visited:
                        continue

                    # D6: Classify the loss for this sub-fragmentation
                    loss = compute_loss(frag, f2)
                    if loss is not None:
                        loss_is_radical = is_radical_species(loss)
                    else:
                        # Direct cation → produced by radical loss
                        loss_is_radical = True

                    # Even-electron rule: EE+ CANNOT lose a radical
                    if USE_EVEN_ELECTRON_RULE and parent_state == "EE" and loss_is_radical:
                        continue   # forbidden transition — skip

                    # Determine child electron state
                    if parent_state == "OE":
                        child_state = "EE" if loss_is_radical else "OE"
                    else:
                        # Parent is EE, only neutral losses allowed → stays EE
                        child_state = "EE"

                    electron_state[key] = child_state
                    visited.add(key)
                    results[key] = (f2, rule2, depth)
                    next_level.append((f2, rule2))

            current_level = next_level

            if not current_level:
                break

        return list(results.values())

    # Public API for HybridEnumerator
    def fragment_formula(self, formula: Formula, start_depth: int = 1):
        fragger = RecursiveFragmenter(formula, self.rule_flags, max_depth=self.max_depth)
        return fragger.generate(start_depth=start_depth)

