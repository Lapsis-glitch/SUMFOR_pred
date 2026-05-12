"""
conformal_calibrator.py

Split (cross-)conformal prediction layer on top of FinalDecisionCalibrator.

The calibration table is a frozen snapshot of (is_correct, oof_prob, stratum)
triples produced by `build_conformal_calibration.py` from the train-pool
K-fold OOF probabilities of the final-decision calibrator. Because every OOF
probability came from a fold model that did not see that entry's group, the
calibration set is exchangeable with predictions on unseen entries (cross-
conformal predictor, Vovk 2015).

Public surface
--------------
- ``ConformalCalibrator(artifact_path).p_value(decision_prob, parent_classes)``
  returns a dict with global and Mondrian-stratified one-sided p-values for the
  null hypothesis "this fragment is a true positive". Small p ⇒ the fragment
  looks worse than calibration TPs ⇒ evidence against the TP hypothesis.

- ``combine_p_values(p_values, method)`` aggregates a list of per-fragment
  p-values into a single spectrum-level p-value. Supported methods:
    * "fisher": classic Fisher combined test (assumes independence)
    * "hmp":    harmonic-mean p (Wilson 2019; robust under positive dependence)
    * "bonferroni_min": ``min(1, k * min(p))`` (valid under arbitrary dependence)
    * "min":   raw minimum (no correction; for diagnostics)

- ``parent_class_group(parent_classes)`` maps the validator's parent-class set
  to a single Mondrian stratum, matching the ordering used by
  ``_resolve_class_profile`` in the precision validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np

ARTIFACT_PATH = "conformal_calibration.pkl"

# Smallest Mondrian stratum size for which we use a stratum-specific calibration
# distribution. Below this we fall back to the global pooled distribution so we
# never return a p-value built from a handful of rows.
MIN_STRATUM_N = 50

# Order matters: stricter chemistry classes win over softer ones, mirroring the
# stricter-wins logic in `_resolve_class_profile` of the precision validator.
CLASS_GROUPS: tuple[str, ...] = (
    "halogenated",
    "sulfur",
    "phosphorus",
    "aromatic",
    "nitrogenous",
    "oxygenated",
)


def parent_class_group(parent_classes: Iterable[str] | None) -> str:
    """Pick the canonical Mondrian stratum for a parent's class set."""
    classes = {str(c) for c in (parent_classes or [])}
    for group in CLASS_GROUPS:
        if group in classes:
            return group
    return "other"


def _empirical_right_tail_p(sorted_scores: np.ndarray, s_new: float, n: int) -> float:
    """Right-tail empirical p with the standard +1/(n+1) plug-in.

    Counts calibration scores ``>= s_new`` (i.e., at least as nonconforming as
    the new point) and applies the conformal smoothing so the returned p is
    never zero.
    """
    if n <= 0:
        return 1.0
    idx = int(np.searchsorted(sorted_scores, s_new, side="left"))
    k = sorted_scores.size - idx
    return (k + 1) / (n + 1)


def combine_p_values(p_values: Sequence[float], method: str = "hmp") -> float:
    """Aggregate a list of per-fragment p-values into one spectrum-level p."""
    arr = np.asarray([p for p in p_values if p is not None and np.isfinite(p)], dtype=float)
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, 1e-12, 1.0)
    k = arr.size

    if method == "min":
        return float(arr.min())
    if method == "bonferroni_min":
        return float(min(1.0, k * arr.min()))
    if method == "fisher":
        chi2 = -2.0 * np.log(arr).sum()
        df = 2 * k
        try:
            from scipy.stats import chi2 as _chi2
            return float(_chi2.sf(chi2, df))
        except ImportError:
            return _wilson_hilferty_chi2_sf(chi2, df)
    if method == "hmp":
        # Asymptotic harmonic-mean p; tighter under positive dependence than Fisher.
        return float(k / np.sum(1.0 / arr))
    raise ValueError(f"Unknown combine method: {method!r}")


def _wilson_hilferty_chi2_sf(x: float, df: int) -> float:
    """Wilson–Hilferty approximation to the chi-square survival function.

    Used only as a fallback when scipy is unavailable. Accurate to a few
    percent for df ≥ 2, which covers our use case (df = 2k, k ≥ 1).
    """
    if df <= 0 or x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / np.sqrt(2.0 / (9.0 * df))
    # 1 - Φ(z) via erfc
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class ConformalCalibrator:
    """Frozen calibration set + p-value queries.

    The calibration artifact stores parallel arrays:
      - ``y``        — int, 0/1, fragment is_correct flag
      - ``prob``     — float, OOF predicted probability of being correct
      - ``stratum``  — str, Mondrian stratum key (parent_class_group output)
      - ``entry_id`` — str, source entry id (kept for traceability)

    Only the positive (``y == 1``) calibration rows are used to test the TP
    null. We sort their nonconformity scores ``s = 1 − prob`` once at load time
    and run binary-search right-tail lookups at inference.
    """

    def __init__(self, artifact_path: str | Path = ARTIFACT_PATH):
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Conformal calibration artifact not found: {path}")
        artifact = joblib.load(path)
        self.artifact_path = path
        self.metadata = dict(artifact.get("metadata", {}))

        y = np.asarray(artifact["y"], dtype=int)
        prob = np.asarray(artifact["prob"], dtype=float)
        stratum = np.asarray(artifact.get("stratum", np.array(["other"] * y.size)), dtype=object)
        if not (y.shape == prob.shape == stratum.shape):
            raise ValueError("Calibration artifact arrays have mismatched shapes.")

        pos_mask = y == 1
        pos_scores_all = 1.0 - prob[pos_mask]
        pos_strata_all = stratum[pos_mask]

        self._global_scores = np.sort(pos_scores_all)
        self._n_global = int(self._global_scores.size)

        self._mondrian_scores: dict[str, np.ndarray] = {}
        for key in np.unique(pos_strata_all):
            mask = pos_strata_all == key
            if int(mask.sum()) < MIN_STRATUM_N:
                continue
            self._mondrian_scores[str(key)] = np.sort(pos_scores_all[mask])

        self._n_total = int(y.size)
        self._n_positive = int(pos_mask.sum())

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    def p_value_global(self, decision_prob: float) -> float:
        """Pooled (marginal) conformal p-value for the TP null."""
        return _empirical_right_tail_p(
            self._global_scores, 1.0 - float(decision_prob), self._n_global
        )

    def p_value_mondrian(self, decision_prob: float, stratum: str) -> tuple[float, str, int]:
        """Class-conditional p-value with auto-fallback to the global pool.

        Returns ``(p, stratum_used, n_calibration_rows_in_stratum)`` where
        ``stratum_used`` is ``"global_fallback"`` when the requested stratum is
        below MIN_STRATUM_N.
        """
        scores = self._mondrian_scores.get(str(stratum))
        if scores is None:
            return self.p_value_global(decision_prob), "global_fallback", self._n_global
        s_new = 1.0 - float(decision_prob)
        return _empirical_right_tail_p(scores, s_new, scores.size), str(stratum), int(scores.size)

    def p_value(self, decision_prob: float | None, parent_classes: Iterable[str] | None) -> dict:
        """Return both global and Mondrian p-values + housekeeping fields.

        A ``None`` decision_prob is treated as "no information available" and
        returns p = 1.0 (cannot reject the TP null).
        """
        if decision_prob is None or not np.isfinite(decision_prob):
            return {
                "p_global": 1.0,
                "p_mondrian": 1.0,
                "stratum_requested": parent_class_group(parent_classes),
                "stratum_used": "no_decision_prob",
                "n_calibration_stratum": 0,
                "n_calibration_global": self._n_global,
            }
        stratum = parent_class_group(parent_classes)
        p_g = self.p_value_global(decision_prob)
        p_m, used, n_strat = self.p_value_mondrian(decision_prob, stratum)
        return {
            "p_global": float(p_g),
            "p_mondrian": float(p_m),
            "stratum_requested": stratum,
            "stratum_used": used,
            "n_calibration_stratum": int(n_strat),
            "n_calibration_global": self._n_global,
        }

    # ------------------------------------------------------------------ #
    # Reporting                                                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        return {
            "artifact_path": str(self.artifact_path),
            "n_calibration_total": self._n_total,
            "n_calibration_positive": self._n_positive,
            "n_global": self._n_global,
            "mondrian_strata": {k: int(v.size) for k, v in self._mondrian_scores.items()},
            "metadata": self.metadata,
        }
