"""
final_decision_calibrator.py

Small calibrated final-decision layer for top1-per-peak fragment export.

This module intentionally stays lightweight:
- stable hand-chosen features only
- logistic regression + optional isotonic calibration
- shared feature mapping for both training and inference

It is meant to refine the final export decision while preserving the main
fragmentation / scoring pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

ARTIFACT_PATH = "final_decision_calibrator.pkl"

CLASS_FLAGS = [
    "aromatic",
    "nitrogenous",
    "oxygenated",
    "halogenated",
    "sulfur",
    "phosphorus",
]

RULE_FLAGS = [
    "bde",
    "neutral",
    "rearrangement",
    "double",
    "aromatic",
    "amine",
    "amide",
    "nitrile",
    "halogen",
    "sulfur",
    "phosphorus",
]

NUMERIC_FEATURES = [
    "ml_prob",
    "hybrid_score",
    "physics_score",
    "peak_posterior",
    "source_peak_rel_intensity",
    "acceptance_score",
    "acceptance_base_score",
    "rule_family_penalty",
    "ambiguity_penalty",
    "n_candidates_for_peak",
    "gap_to_runner_up",
    "mass_defect_abs",
    "mass_defect_norm",
    "neutral_loss_plausible",
    "isotope_consistency",
    "dbe_distance_to_parent",
]

# Neutral defaults for the new chemistry features when records pre-date them.
_FEATURE_DEFAULTS = {
    "mass_defect_abs": 0.0,
    "mass_defect_norm": 0.0,
    "neutral_loss_plausible": 0.5,
    "isotope_consistency": 1.0,
    "dbe_distance_to_parent": 0.0,
}

FEATURE_NAMES = NUMERIC_FEATURES + [f"class_{c}" for c in CLASS_FLAGS] + [f"rule_{r}" for r in RULE_FLAGS]


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        if ";" in text:
            return [part.strip() for part in text.split(";") if part.strip()]
        return [text]
    return [str(value)]


def build_feature_dict_from_record(record: dict) -> dict:
    """Build model features from a saved fragment_metrics.jsonl row."""
    parent_classes = set(_coerce_string_list(record.get("parent_classes")))
    rule_families = set(_coerce_string_list(record.get("rule_families")))

    feat = {
        "ml_prob": _safe_float(record.get("ml_prob")),
        "hybrid_score": _safe_float(record.get("hybrid_score")),
        "physics_score": _safe_float(record.get("physics_score")),
        "peak_posterior": _safe_float(record.get("peak_posterior")),
        "source_peak_rel_intensity": _safe_float(record.get("source_peak_rel_intensity")),
        "acceptance_score": _safe_float(record.get("acceptance_score")),
        "acceptance_base_score": _safe_float(record.get("acceptance_base_score")),
        "rule_family_penalty": _safe_float(record.get("rule_family_penalty")),
        "ambiguity_penalty": _safe_float(record.get("ambiguity_penalty")),
        "n_candidates_for_peak": _safe_float(record.get("n_candidates_for_peak")),
        "gap_to_runner_up": _safe_float(record.get("gap_to_runner_up")),
    }
    for name, default in _FEATURE_DEFAULTS.items():
        feat[name] = _safe_float(record.get(name), default)

    for cls in CLASS_FLAGS:
        feat[f"class_{cls}"] = 1.0 if cls in parent_classes else 0.0
    for rule in RULE_FLAGS:
        feat[f"rule_{rule}"] = 1.0 if rule in rule_families else 0.0
    return feat


def build_feature_dict_from_candidate(candidate: dict, meta_row: dict, parent_classes: Iterable[str] | None) -> dict:
    """Build model features from a live candidate + selector metadata row."""
    parent_classes = set(parent_classes or [])
    rule_families = set(_coerce_string_list(meta_row.get("rule_families")))

    feat = {
        "ml_prob": _safe_float(candidate.get("ml_prob")),
        "hybrid_score": _safe_float(candidate.get("hybrid_score")),
        "physics_score": _safe_float(candidate.get("score")),
        "peak_posterior": _safe_float(meta_row.get("peak_posterior")),
        "source_peak_rel_intensity": _safe_float(candidate.get("source_peak_rel_intensity")),
        "acceptance_score": _safe_float(meta_row.get("acceptance_score")),
        "acceptance_base_score": _safe_float(meta_row.get("acceptance_base_score")),
        "rule_family_penalty": _safe_float(meta_row.get("rule_family_penalty")),
        "ambiguity_penalty": _safe_float(meta_row.get("ambiguity_penalty")),
        "n_candidates_for_peak": _safe_float(meta_row.get("n_candidates_for_peak")),
        "gap_to_runner_up": _safe_float(meta_row.get("gap_to_runner_up")),
    }
    for name, default in _FEATURE_DEFAULTS.items():
        feat[name] = _safe_float(candidate.get(name), default)

    for cls in CLASS_FLAGS:
        feat[f"class_{cls}"] = 1.0 if cls in parent_classes else 0.0
    for rule in RULE_FLAGS:
        feat[f"rule_{rule}"] = 1.0 if rule in rule_families else 0.0
    return feat


def feature_matrix_from_dicts(feature_dicts: list[dict], feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    return np.array([[float(fd.get(name, 0.0)) for name in names] for fd in feature_dicts], dtype=float)


class FinalDecisionCalibrator:
    """Load a tiny final-decision artifact and score top1 candidates."""

    def __init__(self, artifact_path: str | Path = ARTIFACT_PATH):
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Final decision calibrator not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        self.artifact_path = artifact_path
        self.model = artifact["model"]
        self.isotonic = artifact.get("isotonic")
        self.feature_names = artifact.get("feature_names", FEATURE_NAMES)
        self.metadata = artifact.get("metadata", {})

    def predict_proba_from_feature_dict(self, feature_dict: dict) -> float:
        X = feature_matrix_from_dicts([feature_dict], self.feature_names)
        raw_prob = float(self.model.predict_proba(X)[:, 1][0])
        if self.isotonic is not None:
            return float(self.isotonic.predict([raw_prob])[0])
        return raw_prob

    def predict_proba(self, candidate: dict, meta_row: dict, parent_classes: Iterable[str] | None = None) -> float:
        feature_dict = build_feature_dict_from_candidate(candidate, meta_row, parent_classes)
        return self.predict_proba_from_feature_dict(feature_dict)

