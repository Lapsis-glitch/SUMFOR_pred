# fragmentation_rules/registry.py

from __future__ import annotations
from typing import Dict, List, Tuple
from formula import Formula

# Import all rule packs
from . import universal
from . import alcohol
from . import carbonyl
from . import aromatic
from . import amine
from . import ester
from . import halogen
from . import ether
from . import alkene
from . import sulfur
from . import phosphorus


# ------------------------------------------------------------
# Ordered list of rule packs with associated flag keys
# ------------------------------------------------------------
# Order matters: universal rules first, then functional groups.
# Each entry is (module, flag_key).
# flag_key=None means the pack is always enabled.
RULE_PACKS = [
    (universal,    None),
    (alcohol,      "alcohol"),
    (carbonyl,     "carbonyl"),
    (aromatic,     "aromatic"),
    (amine,        "amine"),
    (ester,        "ester"),
    (halogen,      "halogen"),
    (ether,        "ether"),
    (alkene,       "alkene"),
    (sulfur,       "sulfur"),
    (phosphorus,   "phosphorus"),
]


# ------------------------------------------------------------
# Dispatcher (with A4 flag filtering and D2 deduplication)
# ------------------------------------------------------------
def generate_fragments(
    parent: Formula,
    fg: dict,
    rule_flags: Dict[str, bool] | None = None,
) -> List[Tuple[Formula, str]]:
    """
    Dispatch to all rule packs and collect fragments.
    Each rule pack exposes a `generate(parent, fg)` function.

    Parameters
    ----------
    parent : Formula
        The parent molecular formula.
    fg : dict
        Functional group detection results.
    rule_flags : dict, optional
        Enable/disable entire rule families by key.
        If a key is missing, the pack is enabled by default.
    """
    results: List[Tuple[Formula, str]] = []
    seen: set = set()   # D2: deduplicate by formula string

    for pack, flag_key in RULE_PACKS:
        # A4: Check rule flags at pack level
        if flag_key and rule_flags and not rule_flags.get(flag_key, True):
            continue

        try:
            frags = pack.generate(parent, fg)
            if frags:
                for frag, rule in frags:
                    key = frag.to_string()
                    if key not in seen:
                        seen.add(key)
                        results.append((frag, rule))
        except Exception as e:
            # Fail-safe: rule pack errors should not break the engine
            print(f"[Warning] Rule pack {pack.__name__} failed: {e}")

    return results