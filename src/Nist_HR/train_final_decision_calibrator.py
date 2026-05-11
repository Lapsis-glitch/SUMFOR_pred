"""
train_final_decision_calibrator.py

Train a tiny final-decision calibrator for top1-per-peak fragment export.

Training data source:
- schema-2.0 `fragment_metrics.jsonl` files produced by
  `validate_hybrid_fragment_recovery_precision.py`

Workflow:
1. keep retained top1-per-peak candidates only
2. split by entry_id into train / calibration / validation groups
3. fit L2 logistic regression on stable features
4. optionally fit isotonic calibration on calibration-set probabilities
5. save artifact for use by `final_decision_calibrator.py`

Usage:
    python train_final_decision_calibrator.py
    python train_final_decision_calibrator.py validation_outputs_precision/<run_id>/fragment_metrics.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from final_decision_calibrator import (
    ARTIFACT_PATH,
    FEATURE_NAMES,
    build_feature_dict_from_record,
    feature_matrix_from_dicts,
)

BASE_DIR = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs_precision")
METRICS_OUT = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/final_decision_calibrator_metrics.json")


def _find_latest_fragment_metrics() -> Path | None:
    candidates = sorted(BASE_DIR.glob("*/fragment_metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_split(y_true, prob):
    metrics = {
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "brier": float(brier_score_loss(y_true, prob)) if len(set(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "precision_at_0.90": None,
        "count_at_0.90": 0,
        "precision_at_0.95": None,
        "count_at_0.95": 0,
    }
    for thr in (0.90, 0.95):
        mask = prob >= thr
        key_p = f"precision_at_{thr:.2f}"
        key_n = f"count_at_{thr:.2f}"
        metrics[key_n] = int(mask.sum())
        if mask.any():
            metrics[key_p] = float(np.mean(y_true[mask]))
    return metrics


def main():
    if len(sys.argv) > 1:
        metrics_path = Path(sys.argv[1]).expanduser()
    else:
        metrics_path = _find_latest_fragment_metrics()
        if metrics_path is None:
            raise SystemExit("No fragment_metrics.jsonl found under validation_outputs_precision/")

    if not metrics_path.exists():
        raise SystemExit(f"File not found: {metrics_path}")

    rows = _load_jsonl(metrics_path)
    top1_rows = [
        row for row in rows
        if row.get("rank_within_peak") == 1 and row.get("retained_candidate")
    ]
    if not top1_rows:
        raise SystemExit("No top1-per-peak retained candidates found in fragment metrics.")

    entry_ids = sorted({row["entry_id"] for row in top1_rows})
    train_ids, temp_ids = train_test_split(entry_ids, test_size=0.30, random_state=42)
    cal_ids, val_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    def split_rows(ids):
        id_set = set(ids)
        return [row for row in top1_rows if row["entry_id"] in id_set]

    train_rows = split_rows(train_ids)
    cal_rows = split_rows(cal_ids)
    val_rows = split_rows(val_ids)

    X_train = feature_matrix_from_dicts([build_feature_dict_from_record(r) for r in train_rows], FEATURE_NAMES)
    y_train = np.array([int(bool(r.get("is_correct"))) for r in train_rows], dtype=int)

    X_cal = feature_matrix_from_dicts([build_feature_dict_from_record(r) for r in cal_rows], FEATURE_NAMES)
    y_cal = np.array([int(bool(r.get("is_correct"))) for r in cal_rows], dtype=int)

    X_val = feature_matrix_from_dicts([build_feature_dict_from_record(r) for r in val_rows], FEATURE_NAMES)
    y_val = np.array([int(bool(r.get("is_correct"))) for r in val_rows], dtype=int)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)

    cal_raw = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    val_raw = model.predict_proba(X_val)[:, 1]
    val_iso = iso.predict(val_raw)

    raw_metrics = _score_split(y_val, val_raw)
    iso_metrics = _score_split(y_val, val_iso)

    use_isotonic = False
    raw_brier = raw_metrics.get("brier")
    iso_brier = iso_metrics.get("brier")
    if raw_brier is not None and iso_brier is not None and iso_brier <= raw_brier:
        use_isotonic = True

    artifact = {
        "model": model,
        "isotonic": iso if use_isotonic else None,
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "source_metrics": str(metrics_path),
            "n_rows_total": len(top1_rows),
            "n_train": len(train_rows),
            "n_cal": len(cal_rows),
            "n_val": len(val_rows),
            "use_isotonic": use_isotonic,
            "raw_metrics": raw_metrics,
            "isotonic_metrics": iso_metrics,
        },
    }
    artifact_path = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR") / ARTIFACT_PATH
    joblib.dump(artifact, artifact_path)

    report = {
        "artifact_path": str(artifact_path),
        "source_metrics": str(metrics_path),
        "n_top1_rows": len(top1_rows),
        "n_train": len(train_rows),
        "n_cal": len(cal_rows),
        "n_val": len(val_rows),
        "feature_names": FEATURE_NAMES,
        "use_isotonic": use_isotonic,
        "raw_metrics": raw_metrics,
        "isotonic_metrics": iso_metrics,
    }
    with METRICS_OUT.open("w") as f:
        json.dump(report, f, indent=2)

    print("=== Final decision calibrator training ===")
    print(f"source fragment metrics : {metrics_path}")
    print(f"top1 rows used          : {len(top1_rows)}")
    print(f"train/cal/val rows      : {len(train_rows)} / {len(cal_rows)} / {len(val_rows)}")
    print(f"artifact                : {artifact_path}")
    print(f"metrics                 : {METRICS_OUT}")
    print("\nValidation metrics (raw logistic):")
    for k, v in raw_metrics.items():
        print(f"  {k}: {v}")
    print("\nValidation metrics (isotonic):")
    for k, v in iso_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nUsing isotonic: {use_isotonic}")


if __name__ == "__main__":
    main()

