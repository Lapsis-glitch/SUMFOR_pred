"""
train_final_decision_calibrator.py

Train the final-decision calibrator for top1-per-peak fragment export.

Training data source:
- schema-2.0 `fragment_metrics.jsonl` files produced by
  `validate_hybrid_fragment_recovery_precision.py`

Workflow:
1. keep retained top1-per-peak candidates only
2. carve a held-out validation slice by entry_id (default 15%)
3. run stratified group K-fold CV on the remaining training pool
   (group=entry_id, stratify=is_correct) and report per-fold metrics
4. fit the final model on the full training pool
5. evaluate on the held-out validation slice
6. optionally fit isotonic calibration on the CV out-of-fold probabilities
7. save artifact for use by `final_decision_calibrator.py`

The underlying model can be either ``logistic`` (sklearn LogisticRegression,
original behavior) or ``lightgbm`` (LGBMClassifier, default). LightGBM
captures non-linear interactions between the new chemistry features and
the existing acceptance signals — necessary to push validation precision
past the logistic ceiling.

Usage:
    python train_final_decision_calibrator.py
    python train_final_decision_calibrator.py validation_outputs_precision/<run_id>/fragment_metrics.jsonl
    python train_final_decision_calibrator.py --model logistic --no-isotonic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    _HAS_LIGHTGBM = False

from final_decision_calibrator import (
    ARTIFACT_PATH,
    FEATURE_NAMES,
    build_feature_dict_from_record,
    feature_matrix_from_dicts,
)

BASE_DIR = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs_precision")
DEFAULT_METRICS_OUT = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/final_decision_calibrator_metrics.json")
EXTENDED_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)


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


def _score_split(y_true, prob, thresholds=EXTENDED_THRESHOLDS):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    metrics = {
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "brier": float(brier_score_loss(y_true, prob)) if len(set(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None,
    }
    for thr in thresholds:
        mask = prob >= thr
        key_p = f"precision_at_{thr:.2f}"
        key_n = f"count_at_{thr:.2f}"
        metrics[key_n] = int(mask.sum())
        metrics[key_p] = float(np.mean(y_true[mask])) if mask.any() else None
    return metrics


def _build_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
    if kind == "lightgbm":
        if not _HAS_LIGHTGBM:
            raise SystemExit("lightgbm not installed; install lightgbm or use --model logistic.")
        return LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    raise SystemExit(f"Unknown model kind: {kind}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SUMFOR final-decision calibrator.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to fragment_metrics.jsonl. Defaults to the most recent one under validation_outputs_precision/.",
    )
    parser.add_argument(
        "--output-artifact",
        default=None,
        help=f"Where to save the trained calibrator pkl. Defaults to {ARTIFACT_PATH}.",
    )
    parser.add_argument(
        "--metrics-out",
        default=str(DEFAULT_METRICS_OUT),
        help="Where to save the training metrics json.",
    )
    parser.add_argument(
        "--no-isotonic",
        action="store_true",
        help="Force-disable isotonic post-calibration.",
    )
    parser.add_argument(
        "--model",
        choices=("logistic", "lightgbm"),
        default="lightgbm" if _HAS_LIGHTGBM else "logistic",
        help="Underlying classifier (default: lightgbm).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="K for StratifiedGroupKFold.",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.15,
        help="Fraction of entry_ids reserved for the final held-out validation slice.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.input:
        metrics_path = Path(args.input).expanduser()
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
    pool_ids, val_ids = train_test_split(
        entry_ids, test_size=args.holdout_frac, random_state=42
    )
    pool_set = set(pool_ids)
    val_set = set(val_ids)

    pool_rows = [row for row in top1_rows if row["entry_id"] in pool_set]
    val_rows = [row for row in top1_rows if row["entry_id"] in val_set]

    X_pool = feature_matrix_from_dicts(
        [build_feature_dict_from_record(r) for r in pool_rows], FEATURE_NAMES
    )
    y_pool = np.array([int(bool(r.get("is_correct"))) for r in pool_rows], dtype=int)
    groups_pool = np.array([r["entry_id"] for r in pool_rows])

    X_val = feature_matrix_from_dicts(
        [build_feature_dict_from_record(r) for r in val_rows], FEATURE_NAMES
    )
    y_val = np.array([int(bool(r.get("is_correct"))) for r in val_rows], dtype=int)

    # ---- Stratified group K-fold CV on the training pool ---------------
    skf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y_pool), dtype=float)
    fold_metrics = []
    fold_models = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(X_pool, y_pool, groups=groups_pool)
    ):
        model = _build_model(args.model)
        if args.model == "lightgbm":
            model.fit(
                X_pool[tr_idx],
                y_pool[tr_idx],
                eval_set=[(X_pool[va_idx], y_pool[va_idx])],
                eval_metric="average_precision",
                callbacks=[],
            )
        else:
            model.fit(X_pool[tr_idx], y_pool[tr_idx])

        fold_prob = model.predict_proba(X_pool[va_idx])[:, 1]
        oof_prob[va_idx] = fold_prob
        fm = _score_split(y_pool[va_idx], fold_prob)
        fm["fold"] = fold_idx
        fm["n_train"] = int(len(tr_idx))
        fm["n_valid"] = int(len(va_idx))
        fold_metrics.append(fm)
        fold_models.append(model)

    # ---- Train the final model on the full training pool ---------------
    final_model = _build_model(args.model)
    final_model.fit(X_pool, y_pool)

    # ---- Optional isotonic on OOF probabilities ------------------------
    iso = None
    iso_metrics_oof = None
    if not args.no_isotonic:
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(oof_prob, y_pool)
        iso_metrics_oof = _score_split(y_pool, iso.predict(oof_prob))

    raw_metrics_oof = _score_split(y_pool, oof_prob)

    val_raw = final_model.predict_proba(X_val)[:, 1]
    val_iso = iso.predict(val_raw) if iso is not None else None
    raw_metrics_val = _score_split(y_val, val_raw)
    iso_metrics_val = _score_split(y_val, val_iso) if val_iso is not None else None

    # Decide whether to ship isotonic at inference. We default to raw because
    # isotonic flattens the top of the curve, which is where we operate.
    use_isotonic = False
    if iso is not None:
        raw_b = raw_metrics_oof.get("brier")
        iso_b = iso_metrics_oof.get("brier")
        if raw_b is not None and iso_b is not None and iso_b < raw_b - 0.005:
            use_isotonic = True
    if args.no_isotonic:
        use_isotonic = False

    artifact = {
        "model": final_model,
        "isotonic": iso if use_isotonic else None,
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "source_metrics": str(metrics_path),
            "model_kind": args.model,
            "n_top1_rows": len(top1_rows),
            "n_pool_rows": len(pool_rows),
            "n_val_rows": len(val_rows),
            "n_splits": args.n_splits,
            "holdout_frac": args.holdout_frac,
            "use_isotonic": use_isotonic,
            "force_no_isotonic": bool(args.no_isotonic),
            "fold_metrics": fold_metrics,
            "oof_raw_metrics": raw_metrics_oof,
            "oof_isotonic_metrics": iso_metrics_oof,
            "val_raw_metrics": raw_metrics_val,
            "val_isotonic_metrics": iso_metrics_val,
        },
    }
    artifact_path = (
        Path(args.output_artifact).expanduser()
        if args.output_artifact
        else Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR") / ARTIFACT_PATH
    )
    joblib.dump(artifact, artifact_path)

    report = {
        "artifact_path": str(artifact_path),
        "source_metrics": str(metrics_path),
        "model_kind": args.model,
        "n_top1_rows": len(top1_rows),
        "n_pool_rows": len(pool_rows),
        "n_val_rows": len(val_rows),
        "n_splits": args.n_splits,
        "feature_names": FEATURE_NAMES,
        "use_isotonic": use_isotonic,
        "force_no_isotonic": bool(args.no_isotonic),
        "fold_metrics": fold_metrics,
        "oof_raw_metrics": raw_metrics_oof,
        "oof_isotonic_metrics": iso_metrics_oof,
        "val_raw_metrics": raw_metrics_val,
        "val_isotonic_metrics": iso_metrics_val,
    }
    metrics_out_path = Path(args.metrics_out).expanduser()
    with metrics_out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("=== Final decision calibrator training ===")
    print(f"source fragment metrics : {metrics_path}")
    print(f"model kind              : {args.model}")
    print(f"top1 rows used          : {len(top1_rows)}")
    print(f"pool/holdout-val rows   : {len(pool_rows)} / {len(val_rows)}")
    print(f"CV folds                : {args.n_splits}")
    print(f"artifact                : {artifact_path}")
    print(f"metrics                 : {metrics_out_path}")

    print("\nPer-fold CV metrics (AP / Brier / precision@0.85 / precision@0.90):")
    for fm in fold_metrics:
        print(
            f"  fold {fm['fold']}: AP={fm['average_precision']:.4f}  "
            f"Brier={fm['brier']:.4f}  "
            f"P@0.85={fm.get('precision_at_0.85')}  "
            f"P@0.90={fm.get('precision_at_0.90')}  "
            f"P@0.95={fm.get('precision_at_0.95')}"
        )

    print("\nOOF metrics (pooled across folds, raw):")
    for k, v in raw_metrics_oof.items():
        print(f"  {k}: {v}")

    print("\nHeld-out validation metrics (final model, raw):")
    for k, v in raw_metrics_val.items():
        print(f"  {k}: {v}")

    if iso_metrics_val is not None:
        print("\nHeld-out validation metrics (final model, isotonic):")
        for k, v in iso_metrics_val.items():
            print(f"  {k}: {v}")

    print(f"\nUsing isotonic at inference: {use_isotonic}")


if __name__ == "__main__":
    main()
