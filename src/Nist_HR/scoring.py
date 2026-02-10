"""
scoring.py — pure scoring, no correctness logic

Computes:
    - physics-only score
    - ML-only score
    - hybrid score

Does NOT:
    - compare to AML peaks
    - determine correctness
    - filter by confidence
"""

from __future__ import annotations
from typing import List


def score_assignments(assignments: List[dict],
                      score_key: str = "score") -> List[dict]:
    """
    Attach the chosen score to each assignment.

    Parameters
    ----------
    assignments : list of dict
        Must contain:
            - "score" (physics)
            - "ml_prob" (ML)
            - "hybrid_score" (physics × ML)

    score_key : str
        "score"          → physics-only
        "ml_prob"        → ML-only
        "hybrid_score"   → physics × ML

    Returns
    -------
    list of dict
        Each dict gets a new field:
            - "used_score"
    """

    out = []
    for a in assignments:
        a_out = dict(a)
        a_out["used_score"] = a.get(score_key, 0.0)
        out.append(a_out)

    return out