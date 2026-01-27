"""
fps_scoring.py

Fragment Plausibility Score (FPS) for EI fragments.

Now supports tunable weights for:
- feature contributions
- rule-family priors
"""

from __future__ import annotations
from typing import Iterable, Dict
from formula import Formula
from chemistry import exact_mass, dbe


# ============================================================
# Tunable global weights
# ============================================================

FPS_WEIGHTS: Dict[str, float] = {
    # feature weights
    "w_cation":        0.25,
    "w_hetero":        0.20,
    "w_aromatic":      0.15,
    "w_masspos":       0.15,
    "w_context":       0.15,
    "w_prior":         0.10,
}

# Rule-family priors (also tunable)
RULE_FAMILY_PRIORS: Dict[str, float] = {
    "neutral_loss":        0.5,
    "double_loss":         0.3,
    "cation":              0.7,
    "alpha":               0.6,
    "rearrangement":       0.4,
    "oxygen_adjacent":     0.6,
    "hydrogen_transfer":   0.3,
    "mclafferty":          0.7,

    "alcohol_beta":        0.7,
    "alcohol_dehydration": 0.8,
    "alcohol_gamma_shift": 0.5,

    "acylium":             0.8,
    "carbonyl_alpha":      0.7,

    "tropylium":           0.9,
    "phenyl":              0.8,
    "benzyl":              0.8,
    "ring_contraction":    0.5,

    "iminium":             0.7,
    "amine_alpha":         0.6,

    "ester_acylium":       0.7,
    "ester_alkoxy":        0.5,

    "halogen_cation":      0.8,
    "halogen_loss":        0.6,

    "ether_alpha":         0.6,
    "allylic":             0.7,
}


# ============================================================
# Feature helpers
# ============================================================

def _cation_stability(frag: Formula) -> float:
    elems = frag.elements
    c = elems.get("C", 0)
    h = elems.get("H", 0)
    rdbe = dbe(frag)

    score = 0.0

    if 1 <= c <= 4 and 3 <= h <= 10:
        score += 0.5

    if c >= 6 and rdbe >= 4:
        score += 0.4

    return min(score, 1.0)


def _heteroatom_stabilization(frag: Formula) -> float:
    elems = frag.elements
    score = 0.0

    if "O" in elems:
        score += 0.3
    if "N" in elems:
        score += 0.3
    if any(x in elems for x in ["Cl", "Br", "F", "I"]):
        score += 0.2

    return min(score, 1.0)


def _aromatic_stabilization(frag: Formula) -> float:
    elems = frag.elements
    c = elems.get("C", 0)
    rdbe = dbe(frag)

    if c >= 6 and rdbe >= 4:
        return 0.7
    return 0.0


def _mass_position_factor(frag: Formula, parent: Formula) -> float:
    m_frag = exact_mass(frag)
    m_parent = exact_mass(parent)

    if m_frag < 20:
        return 0.1
    if m_frag > 0.9 * m_parent:
        return 0.2
    if 0.2 * m_parent <= m_frag <= 0.8 * m_parent:
        return 0.6
    return 0.4


def _spectrum_context_factor(frag: Formula,
                             observed_nominals: Iterable[int]) -> float:
    obs = set(observed_nominals)
    nm = round(exact_mass(frag))

    score = 0.0
    if nm in obs:
        score += 0.4
    if (nm - 14) in obs or (nm + 14) in obs:
        score += 0.2

    return min(score, 1.0)


# ============================================================
# Main FPS function (now weightable)
# ============================================================

def fragment_plausibility_score(
    frag: Formula,
    parent: Formula,
    observed_nominals: Iterable[int],
    rule_source: str | None,
) -> float:

    # compute feature values
    cation = _cation_stability(frag)
    hetero = _heteroatom_stabilization(frag)
    arom = _aromatic_stabilization(frag)
    mass_pos = _mass_position_factor(frag, parent)
    context = _spectrum_context_factor(frag, observed_nominals)

    # rule prior
    prior = RULE_FAMILY_PRIORS.get(rule_source or "", 0.4)

    # weighted combination
    score = (
        FPS_WEIGHTS["w_cation"]   * cation +
        FPS_WEIGHTS["w_hetero"]   * hetero +
        FPS_WEIGHTS["w_aromatic"] * arom +
        FPS_WEIGHTS["w_masspos"]  * mass_pos +
        FPS_WEIGHTS["w_context"]  * context +
        FPS_WEIGHTS["w_prior"]    * prior
    )

    return max(0.0, min(score, 1.0))