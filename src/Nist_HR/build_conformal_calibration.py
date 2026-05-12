"""
build_conformal_calibration.py

Build the conformal calibration table used by ``conformal_calibrator.py``.

The table is produced by stratified group K-fold cross-validation over the
final-decision calibrator on the train-pool's top1-per-peak rows. Each row's
``oof_prob`` comes from a fold model that was not trained on its entry's group,
so the resulting (y, oof_prob, stratum) triples are exchangeable with
predictions on unseen entries — the cross-conformal predictor of Vovk (2015).

Input
-----
A ``fragment_metrics.jsonl`` produced by a ``train_pool`` run of
``validate_hybrid_fragment_recovery_precision.py``. By default, the most
recent such file under ``validation_outputs_precision/`` is used.

Output
------
A joblib pickle (``conformal_calibration.pkl`` by default) containing parallel
numpy arrays ``y``, ``prob``, ``stratum``, ``entry_id``, plus diagnostic
metadata. The default location sits next to ``final_decision_calibrator.pkl``
so the precision validator can find it without extra config.

Optionally, with ``--verify-on PATH``, the script also evaluates empirical
coverage of the resulting calibration set against a held-out
``fragment_metrics.jsonl`` (typically a ``test_only`` run), and prints a small
table of nominal vs. observed coverage.

Usage
-----
    python build_conformal_calibration.py
    python build_conformal_calibration.py validation_outputs_precision/<train_pool>/fragment_metrics.jsonl
    python build_conformal_calibration.py --verify-on validation_outputs_precision/<test_only>/fragment_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    _HAS_LIGHTGBM = False

from sklearn.linear_model import LogisticRegression

from conformal_calibrator import (
    ARTIFACT_PATH as CONFORMAL_ARTIFACT_NAME,
    MIN_STRATUM_N,
    parent_class_group,
)
from final_decision_calibrator import (
    FEATURE_NAMES,
    build_feature_dict_from_record,
    feature_matrix_from_dicts,
)
from heldout_split import load_heldout_test_ids

BASE_DIR = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR/validation_outputs_precision")
DEFAULT_ARTIFACT = Path("/home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR") / CONFORMAL_ARTIFACT_NAME

NOMINAL_LEVELS = (0.01, 0.05, 0.10, 0.20)


def _find_latest_train_pool_metrics() -> Path | None:
    candidates = sorted(
        BASE_DIR.glob("*train_pool*/fragment_metrics.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = sorted(BASE_DIR.glob("*/fragment_metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fallback[0] if fallback else None


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_model(kind: str, n_jobs: int = 4):
    if kind == "lightgbm":
        if not _HAS_LIGHTGBM:
            raise SystemExit("lightgbm not installed; install lightgbm or pass --model logistic.")
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
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if kind == "logistic":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
    raise SystemExit(f"Unknown model kind: {kind}")


def _filter_top1_retained(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("rank_within_peak") == 1 and r.get("retained_candidate")]


def _drop_heldout(rows: list[dict]) -> list[dict]:
    heldout = load_heldout_test_ids()
    if not heldout:
        return rows
    return [r for r in rows if str(r.get("entry_id")) not in heldout]


def _row_stratum(row: dict) -> str:
    parent_classes = row.get("parent_classes") or []
    if isinstance(parent_classes, str):
        # CSV-ish fallback; the validator usually writes a list directly.
        parent_classes = [c.strip() for c in parent_classes.strip("[]").replace("'", "").split(",") if c.strip()]
    return parent_class_group(parent_classes)


def _oof_predictions(rows: list[dict], model_kind: str, n_splits: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import time as _time
    X = feature_matrix_from_dicts([build_feature_dict_from_record(r) for r in rows], FEATURE_NAMES)
    y = np.array([int(bool(r.get("is_correct"))) for r in rows], dtype=int)
    groups = np.array([str(r.get("entry_id")) for r in rows])
    oof = np.zeros(len(y), dtype=float)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y, groups=groups)):
        t0 = _time.time()
        model = _build_model(model_kind, n_jobs=n_jobs)
        # eval_set is intentionally omitted — we don't use early stopping for the
        # OOF table, and the per-iteration eval overhead dominates runtime.
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        print(f"  fold {fold_idx}: train={len(tr_idx)} val={len(va_idx)} fit_s={_time.time()-t0:.1f}", flush=True)
    return oof, y, groups


def _build_calibration_artifact(rows: list[dict], oof_prob: np.ndarray, y: np.ndarray) -> dict:
    strata = np.array([_row_stratum(r) for r in rows], dtype=object)
    entry_ids = np.array([str(r.get("entry_id")) for r in rows], dtype=object)

    artifact = {
        "y": y.astype(int),
        "prob": oof_prob.astype(float),
        "stratum": strata,
        "entry_id": entry_ids,
        "metadata": {
            "feature_names": FEATURE_NAMES,
            "min_stratum_n": MIN_STRATUM_N,
            "n_rows": int(y.size),
            "n_positive": int(y.sum()),
            "stratum_counts": {
                str(s): int((strata == s).sum()) for s in np.unique(strata)
            },
            "stratum_positive_counts": {
                str(s): int(((strata == s) & (y == 1)).sum()) for s in np.unique(strata)
            },
        },
    }
    return artifact


def _empirical_coverage(test_y: np.ndarray, test_prob: np.ndarray, cal_pos_scores: np.ndarray, levels=NOMINAL_LEVELS) -> dict:
    """For positives in the test set, fraction of conformal p-values <= alpha.

    Under valid conformal calibration, this fraction should be ≤ alpha — i.e.,
    we wrongly reject the TP null at rate ≤ alpha on truly-positive fragments.
    """
    if cal_pos_scores.size == 0:
        return {}
    sorted_cal = np.sort(cal_pos_scores)
    n = sorted_cal.size

    pos_mask = test_y == 1
    test_s = 1.0 - test_prob[pos_mask]
    p_vals = np.array([
        (sorted_cal.size - np.searchsorted(sorted_cal, s, side="left") + 1) / (n + 1)
        for s in test_s
    ])

    report = {
        "n_test_positive": int(test_s.size),
        "p_value_mean": float(p_vals.mean()) if p_vals.size else None,
        "p_value_median": float(np.median(p_vals)) if p_vals.size else None,
    }
    for alpha in levels:
        report[f"empirical_reject_rate_at_alpha_{alpha:.2f}"] = float(np.mean(p_vals <= alpha)) if p_vals.size else None
    return report


def main():
    parser = argparse.ArgumentParser(description="Build the SUMFOR conformal calibration table.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a train_pool fragment_metrics.jsonl. Defaults to the most recent train_pool run.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT),
        help=f"Where to save the conformal calibration pkl (default: {DEFAULT_ARTIFACT}).",
    )
    parser.add_argument(
        "--model",
        choices=("logistic", "lightgbm"),
        default="lightgbm" if _HAS_LIGHTGBM else "logistic",
        help="Underlying classifier used by the K-fold CV (default: lightgbm).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="K for StratifiedGroupKFold (default: 5).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Threads per fold (default: 4). -1 spawns one per core, which contends with OpenMP and often slows things down.",
    )
    parser.add_argument(
        "--verify-on",
        default=None,
        help="Optional: path to a test_only fragment_metrics.jsonl to check empirical coverage.",
    )
    args = parser.parse_args()

    if args.input:
        metrics_path = Path(args.input).expanduser()
    else:
        metrics_path = _find_latest_train_pool_metrics()
        if metrics_path is None:
            raise SystemExit("No train_pool fragment_metrics.jsonl found.")
    if not metrics_path.exists():
        raise SystemExit(f"File not found: {metrics_path}")

    rows = _load_jsonl(metrics_path)
    rows = _filter_top1_retained(rows)
    before = len(rows)
    rows = _drop_heldout(rows)
    print(f"=== Building conformal calibration table ===")
    print(f"  source        : {metrics_path}")
    print(f"  top1 rows     : {before}  (after heldout drop: {len(rows)})")

    print(f"  K-fold CV     : {args.n_splits} folds, model={args.model}, n_jobs={args.n_jobs}")
    oof_prob, y, _groups = _oof_predictions(rows, args.model, args.n_splits, args.n_jobs)

    artifact = _build_calibration_artifact(rows, oof_prob, y)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)

    meta = artifact["metadata"]
    print(f"  rows          : {meta['n_rows']}")
    print(f"  positives     : {meta['n_positive']}")
    print(f"  artifact      : {out_path}")
    print(f"  strata (all)  : {meta['stratum_counts']}")
    print(f"  strata (pos)  : {meta['stratum_positive_counts']}")

    if args.verify_on:
        verify_path = Path(args.verify_on).expanduser()
        if not verify_path.exists():
            raise SystemExit(f"--verify-on file not found: {verify_path}")
        test_rows = _filter_top1_retained(_load_jsonl(verify_path))
        if not test_rows:
            raise SystemExit("No top1-per-peak retained rows in the verify file.")
        test_y = np.array([int(bool(r.get("is_correct"))) for r in test_rows], dtype=int)
        test_prob = np.array(
            [float(r.get("decision_prob") or 0.0) for r in test_rows], dtype=float
        )

        cal_pos = (1.0 - oof_prob)[y == 1]
        coverage = _empirical_coverage(test_y, test_prob, cal_pos)
        print(f"\n=== Empirical coverage on {verify_path} ===")
        for k, v in coverage.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
