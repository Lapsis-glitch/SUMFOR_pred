# peak_driven_assignment_engine.py

from __future__ import annotations
from typing import List, Tuple
import math

from peakassignment import PeakAssignment
from fps_scoring import fragment_plausibility_score
from assignment_engine import AssignmentEngine
from chemistry import exact_mass


USE_FPS = True
USE_MULTI_MATCH = True
USE_INTENSITY_WEIGHT = True

# ============================================================
# Tunable scoring parameters (updated by tune_scoring.py)
# ============================================================
SCORING_PARAMS = {
    "min_raw_score": 0.05,      # below this → no-assignment
    "margin_sharpness": 8.0,    # logistic sharpness
    "depth_penalty": 0.3,       # penalty per depth level
    "intensity_gamma": 1.0,     # exponent for intensity weighting
}


def _logistic(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))


class PeakDrivenAssignmentEngine:
    """
    Peak-driven assignment:
    For each NIST peak, enumerate only fragments that match that m/z.
    """

    def __init__(self, parent_formula, peak_enumerator):
        self.parent_formula = parent_formula
        self.enumerator = peak_enumerator

    def _legacy_score(self, frag, observed_nominals):
        ae = AssignmentEngine(self.parent_formula, [])
        return ae._legacy_conditional_score(frag, observed_nominals)

    # ------------------------------------------------------------
    # Discriminative confidence computation
    # ------------------------------------------------------------
    def _compute_confidences(self, raw_scores: List[float]) -> List[float]:

        if not raw_scores:
            return []

        best_raw = max(raw_scores)
        if best_raw < SCORING_PARAMS["min_raw_score"]:
            return [0.0 for _ in raw_scores]

        # second-best
        sorted_scores = sorted(raw_scores, reverse=True)
        second_raw = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_raw - second_raw

        base_conf = _logistic(
            margin,
            k=SCORING_PARAMS["margin_sharpness"]
        )

        # scale each candidate by closeness to best
        confidences = []
        for s in raw_scores:
            rel = s / best_raw if best_raw > 0 else 0.0
            confidences.append(base_conf * rel)

        return confidences

    # ------------------------------------------------------------
    # Main assignment loop
    # ------------------------------------------------------------
    def assign_peaks(self, nist_peaks, rel_intensities):
        observed_nominals = [mz for mz, _ in nist_peaks]
        assignments = []

        for (nominal_mz, intensity), Irel in zip(nist_peaks, rel_intensities):

            candidates = self.enumerator.enumerate_for_peak(nominal_mz)
            # candidates: [(frag, rule_source, depth), ...]

            # # DEBUG: count BDE vs rule-based fragments before physics filtering
            # bde_candidates = [c for c in candidates if "bde" in c[1]]
            # rule_candidates = [c for c in candidates if "bde" not in c[1]]
            #
            # print(f"[DEBUG] Peak {nominal_mz}:")
            # print(f"  BDE candidates before physics: {len(bde_candidates)}")
            # print(f"  Rule-based candidates before physics: {len(rule_candidates)}")

            if not candidates:
                assignments.append(
                    PeakAssignment(
                        nominal_mz=nominal_mz,
                        intensity=intensity,
                        best_exact_mz=None,
                        best_formula=None,
                        candidate_formulas=[],
                        confidence=0.0,
                        rule_source=None,
                    )
                )
                continue

            raw_scores = []
            for frag, rule_source, depth in candidates:

                # FPS or legacy score
                if USE_FPS:
                    s = fragment_plausibility_score(
                        frag=frag,
                        parent=self.parent_formula,
                        observed_nominals=observed_nominals,
                        rule_source=rule_source,
                    )
                else:
                    s = self._legacy_score(frag, observed_nominals)

                # depth penalty
                s *= 1.0 / (1.0 + SCORING_PARAMS["depth_penalty"] * (depth - 1))

                # intensity weighting
                if USE_INTENSITY_WEIGHT:
                    gamma = SCORING_PARAMS["intensity_gamma"]
                    w = 0.3 + 0.7 * (Irel ** gamma)
                    s *= w

                raw_scores.append(s)

            # discriminative confidence
            conf_scores = self._compute_confidences(raw_scores)

            # # DEBUG: count BDE vs rule-based survivors
            # bde_survivors = 0
            # rule_survivors = 0
            #
            # for (frag, rule_source, depth), conf in zip(candidates, conf_scores):
            #     if conf > 0.0:
            #         if "bde" in rule_source:
            #             bde_survivors += 1
            #         else:
            #             rule_survivors += 1
            #
            # print(f"  BDE survivors after physics: {bde_survivors}")
            # print(f"  Rule-based survivors after physics: {rule_survivors}")
            # print(f"  BDE eliminated by physics: {len(bde_candidates) - bde_survivors}")
            # print()

            # no-assignment case
            if all(c == 0.0 for c in conf_scores):
                assignments.append(
                    PeakAssignment(
                        nominal_mz=nominal_mz,
                        intensity=intensity,
                        best_exact_mz=None,
                        best_formula=None,
                        candidate_formulas=[],
                        confidence=0.0,
                        rule_source="no_assignment",
                    )
                )
                continue

            # normal case
            for (frag, rule_source, depth), conf in zip(candidates, conf_scores):
                assignments.append(
                    PeakAssignment(
                        nominal_mz=nominal_mz,
                        intensity=intensity,
                        best_exact_mz=exact_mass(frag),
                        best_formula=frag.to_string(),
                        candidate_formulas=[frag.to_string()],
                        confidence=conf,
                        rule_source=f"{rule_source} (depth={depth})",
                    )
                )

        return assignments