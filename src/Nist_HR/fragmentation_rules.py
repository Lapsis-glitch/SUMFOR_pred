# fragmentation_rules.py

"""
Modular rule-based fragment generator for EI mass spectrometry.

This file is now a lightweight orchestrator:
- detects functional groups
- dispatches to rule packs via the registry
- applies rule flags
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from formula import Formula

from fragmentation_rules_info.base import detect_functional_groups
from fragmentation_rules_info.registry import generate_fragments


def generate_rule_based_fragments(
    parent: Formula,
    rule_flags: Dict[str, bool] | None = None,
    auto_detect_rules: bool = True,
) -> List[Tuple[Formula, str]]:
    """
    Generate fragments from a parent formula using modular rule packs.

    Parameters
    ----------
    parent : Formula
        The parent molecular formula.
    rule_flags : dict
        Optional dictionary enabling/disabling rule families.
        If None, all rule packs are enabled.
    auto_detect_rules : bool
        If True, detect functional groups automatically.

    Returns
    -------
    List of (fragment_formula, rule_source)
    """

    # Default: all rules enabled
    if rule_flags is None:
        rule_flags = {}

    # Functional group detection
    fg = detect_functional_groups(parent) if auto_detect_rules else {}

    # Dispatch to rule packs
    fragments = generate_fragments(parent, fg)

    # Apply rule flags (post-filter)
    if rule_flags:
        filtered = []
        for frag, source in fragments:
            # If a rule flag is explicitly disabled, skip it
            family = source.split("_")[0]  # e.g. "alcohol", "sulfur", "neutral"
            if family in rule_flags and not rule_flags[family]:
                continue
            filtered.append((frag, source))
        return filtered

    return fragments