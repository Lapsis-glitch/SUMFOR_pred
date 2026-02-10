# fragmentation_rules/registry.py

from __future__ import annotations
from typing import List, Tuple
from src.Nist_HR.formula import Formula

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
# Ordered list of rule packs
# ------------------------------------------------------------
# Order matters: universal rules first, then functional groups.
RULE_PACKS = [
    universal,
    alcohol,
    carbonyl,
    aromatic,
    amine,
    ester,
    halogen,
    ether,
    alkene,
    sulfur,
    phosphorus,
]


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------
def generate_fragments(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """
    Dispatch to all rule packs and collect fragments.
    Each rule pack exposes a `generate(parent, fg)` function.
    """
    results: List[Tuple[Formula, str]] = []

    for pack in RULE_PACKS:
        try:
            frags = pack.generate(parent, fg)
            if frags:
                results.extend(frags)
        except Exception as e:
            # Fail-safe: rule pack errors should not break the engine
            print(f"[Warning] Rule pack {pack.__name__} failed: {e}")

    return results