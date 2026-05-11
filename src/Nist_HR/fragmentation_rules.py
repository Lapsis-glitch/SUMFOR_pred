# fragmentation_rules.py

"""
Modular rule-based fragment generator for EI mass spectrometry.

This file is now a lightweight orchestrator:
- detects functional groups
- dispatches to rule packs via the registry
- applies rule flags (at the pack level, via registry)
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
    parent_fg: Dict[str, bool] | None = None,
) -> List[Tuple[Formula, str]]:
    """
    Generate fragments from a parent formula using modular rule packs.

    Parameters
    ----------
    parent : Formula
        The parent molecular formula.
    rule_flags : dict
        Optional dictionary enabling/disabling rule families.
        Keys should match pack flag keys: "alcohol", "carbonyl",
        "aromatic", "amine", "ester", "halogen", "ether", "alkene",
        "sulfur", "phosphorus".
        If None, all rule packs are enabled.
    auto_detect_rules : bool
        If True, detect functional groups automatically.
    parent_fg : dict, optional
        Pre-computed functional group dict (e.g. from SMARTS-based
        detection). If provided, overrides auto-detection.

    Returns
    -------
    List of (fragment_formula, rule_source)
    """

    # Default: all rules enabled
    if rule_flags is None:
        rule_flags = {}

    # Functional group detection
    if parent_fg is not None:
        fg = parent_fg
    elif auto_detect_rules:
        fg = detect_functional_groups(parent)
    else:
        fg = {}

    # Dispatch to rule packs (flag filtering and deduplication handled by registry)
    return generate_fragments(parent, fg, rule_flags)
