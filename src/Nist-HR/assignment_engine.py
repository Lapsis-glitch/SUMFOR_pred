"""
assignment_engine.py

Assigns NIST integer-m/z peaks to fragment formulas and computes
confidence scores.

Supports:
- Multi-assignment mode (one PeakAssignment per fragment)
- Two scoring models:
    * Legacy conditional scoring (with base_confidence)
    * FPS (Fragment Plausibility Score)
- A flag to switch between scoring models
- Per-peak normalization so both scoring modes produce comparable confidences
"""

from __future__ import annotations
from typing import List, Tuple, Iterable
import numpy as np

from fragment import Fragment
from peakassignment import PeakAssignment

from fps_scoring import fragment_plausibility_score


# ------------------------------------------------------------
# Global flags
# ------------------------------------------------------------
USE_MULTI_MATCH = True     # emit one assignment per fragment
USE_FPS = True            # False = old behavior, True = FPS


class AssignmentEngine:
    """
    Assigns exact masses and formulas to NIST peaks using a list of
    precomputed Fragment objects.

    In multi-match mode:
        For each m/z, emit one PeakAssignment per fragment candidate.

    Scoring:
        - If USE_FPS = True → use FPS scoring
        - Else → use legacy base_confidence + conditional scoring
    """

    def __init__(self, parent_formula, fragments: List[Fragment]):
        self.parent_formula = parent_formula
        self.fragments = fragments
        self.nominal_lookup = self._build_nominal_lookup()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _build_nominal_lookup(self):
        lookup = {}
        for frag in self.fragments:
            lookup.setdefault(frag.nominal_mass, []).append(frag)
        return lookup

    # ---------------- Legacy conditional scoring ---------------

    def _legacy_conditional_score(self, frag: Fragment, observed_nominals: Iterable[int]) -> float:
        """
        Original conditional heuristic (without base_confidence).
        """
        observed_nominals = set(observed_nominals)
        score = 0.0

        # Homologous series: +/- 14 (CH2)
        if frag.nominal_mass - 14 in observed_nominals:
            score += 0.15
        if frag.nominal_mass + 14 in observed_nominals:
            score += 0.15

        # Aromatic hints
        if frag.formula.elements.get("C", 0) >= 6 and frag.dbe >= 4:
            if 77 in observed_nominals or 91 in observed_nominals:
                score += 0.25

        # Common neutral losses: H2O, CO, CO2
        for loss in [18, 28, 44]:
            if frag.nominal_mass + loss in observed_nominals:
                score += 0.2

        # Halogens
        elems = frag.formula.elements
        if "Cl" in elems and (35 in observed_nominals or 37 in observed_nominals):
            score += 0.3
        if "Br" in elems and (79 in observed_nominals or 81 in observed_nominals):
            score += 0.3

        return score

    def _legacy_raw_scores(self, candidates: List[Fragment], observed_nominals: Iterable[int]) -> List[float]:
        """
        Original behavior:
            base_confidence = 1 / n_candidates
            confidence_raw = base_confidence + conditional_score
        Normalization happens later.
        """
        if not candidates:
            return []

        base_conf = 1.0 / len(candidates)
        cond_scores = [self._legacy_conditional_score(f, observed_nominals) for f in candidates]
        raw = [base_conf + cs for cs in cond_scores]
        return raw

    # ---------------- FPS scoring ------------------------------

    def _fps_raw_scores(self, candidates: List[Fragment], observed_nominals: Iterable[int]) -> List[float]:
        """
        FPS-based raw scores (0–1).
        Normalization happens later.
        """
        return [
            fragment_plausibility_score(
                frag=f.formula,
                parent=self.parent_formula,
                observed_nominals=observed_nominals,
                rule_source=f.rule_source,
            )
            for f in candidates
        ]

    # ---------------- Normalization ----------------------------

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores so that:
            max_score → 1.0
            others → proportional
        This restores legacy behavior and makes FPS comparable.
        """
        if not scores:
            return scores
        max_s = max(scores)
        if max_s <= 0:
            return [0.0 for _ in scores]
        return [s / max_s for s in scores]

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def assign_peaks(self, nist_peaks: List[Tuple[int, float]]) -> List[PeakAssignment]:
        observed_nominals = [mz for mz, _ in nist_peaks]
        assignments: List[PeakAssignment] = []

        for nominal_mz, intensity in nist_peaks:
            candidates = self.nominal_lookup.get(nominal_mz, [])

            # Case 1: no candidates
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

            # Compute raw scores according to selected model
            if USE_FPS:
                raw_scores = self._fps_raw_scores(candidates, observed_nominals)
            else:
                raw_scores = self._legacy_raw_scores(candidates, observed_nominals)

            # Normalize to [0, 1], best = 1.0
            conf_scores = self._normalize_scores(raw_scores)

            # ----------------------------------------------------
            # Multi-match mode: emit one assignment per fragment
            # ----------------------------------------------------
            if USE_MULTI_MATCH:
                for frag, conf in zip(candidates, conf_scores):
                    assignments.append(
                        PeakAssignment(
                            nominal_mz=nominal_mz,
                            intensity=intensity,
                            best_exact_mz=frag.exact_mass,
                            best_formula=frag.formula.to_string(),
                            candidate_formulas=[frag.formula.to_string()],
                            confidence=conf,
                            rule_source=frag.rule_source,
                        )
                    )
                continue

            # ----------------------------------------------------
            # Single-best mode (legacy)
            # ----------------------------------------------------
            best_idx = int(np.argmax(conf_scores))
            best_frag = candidates[best_idx]
            best_conf = conf_scores[best_idx]

            assignments.append(
                PeakAssignment(
                    nominal_mz=nominal_mz,
                    intensity=intensity,
                    best_exact_mz=best_frag.exact_mass,
                    best_formula=best_frag.formula.to_string(),
                    candidate_formulas=[best_frag.formula.to_string()],
                    confidence=best_conf,
                    rule_source=best_frag.rule_source,
                )
            )

        return assignments