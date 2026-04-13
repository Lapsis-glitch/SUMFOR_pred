# validate_hybrid_precision_extended.py
import time
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from Large_data import ENTRY_IDS, run_single_entry

TOL = 0.0001
HYBRID_THR = 0.75


RESULTS_OUTPUT = {
    "enabled": True,
    "detail": "entry_and_fragment",  # entry | entry_and_fragment
    "formats": {
        "entry": "csv",       # csv | jsonl
        "fragment": "jsonl",  # csv | jsonl
        "run_summary": "json"
    },
    "out_dir": "validation_outputs",
    "run_name": "auto",
    "flush_every": 100,
    "include_fake_spectra": False,
    "schema_version": "1.0"
}


def is_correct(frag_mz, aml_mz_list, tol=TOL):
    if frag_mz is None:
        return False
    return any(abs(frag_mz - m) <= tol for m in aml_mz_list)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_stats(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    arr = np.array(clean, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min())


def _atomic_write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_jsonl(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _atomic_write_csv(path, rows):
    tmp = f"{path}.tmp"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_rows(path_base, fmt, rows):
    if not rows:
        return None
    if fmt == "csv":
        path = f"{path_base}.csv"
        _atomic_write_csv(path, rows)
        return path
    if fmt == "jsonl":
        path = f"{path_base}.jsonl"
        _atomic_write_jsonl(path, rows)
        return path
    raise ValueError(f"Unsupported format: {fmt}")


def save_validation_results(config, run_id, entry_records, fragment_records, run_summary):
    if not config.get("enabled", False):
        return {}

    run_dir = os.path.join(config.get("out_dir", "validation_outputs"), run_id)
    os.makedirs(run_dir, exist_ok=True)

    written = {}
    entry_format = config.get("formats", {}).get("entry", "csv")
    fragment_format = config.get("formats", {}).get("fragment", "jsonl")
    summary_format = config.get("formats", {}).get("run_summary", "json")

    entry_path = _write_rows(os.path.join(run_dir, "entry_metrics"), entry_format, entry_records)
    if entry_path:
        written["entry"] = entry_path

    if config.get("detail") == "entry_and_fragment":
        fragment_path = _write_rows(os.path.join(run_dir, "fragment_metrics"), fragment_format, fragment_records)
        if fragment_path:
            written["fragment"] = fragment_path

    _atomic_write_json(os.path.join(run_dir, "config_snapshot.json"), config)
    written["config"] = os.path.join(run_dir, "config_snapshot.json")

    if summary_format == "json":
        summary_path = os.path.join(run_dir, "run_summary.json")
        _atomic_write_json(summary_path, run_summary)
        written["run_summary"] = summary_path

    return written


# ------------------------------------------------------------
# Export predicted fragments as a “fake HR spectrum”
# ------------------------------------------------------------
def export_fake_spectrum(entry_id, parent_name, parent_formula, assignments, out_dir="fake_spectra"):
    """
    Saves a JSON file containing:
        - predicted fragment m/z
        - hybrid confidence
        - fragment formula
        - parent name
        - parent formula
    """

    data = {
        "entry_id": entry_id,
        "parent_name": parent_name,
        "parent_formula": parent_formula,
        "predicted_fragments": []
    }

    for a in assignments:
        if a.get("hybrid_score", 0.0) < HYBRID_THR:
            continue

        data["predicted_fragments"].append({
            "mz": a.get("best_exact_mz"),
            "hybrid_score": a.get("hybrid_score"),
            "formula": a.get("best_formula")
        })

    # Save JSON
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/entry_{entry_id}.json"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------
# Main validation
# ------------------------------------------------------------
def main():
    precisions = []
    predicted_counts = []
    correct_counts = []
    entry_records = []
    fragment_records = []

    run_start = time.time()
    run_id = RESULTS_OUTPUT.get("run_name", "auto")
    if run_id == "auto":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Running hybrid precision validation...")
    count = 0
    t = time.time()
    for eid in ENTRY_IDS:
        count +=1
        if count % 100 == 0:
            print(f"Processing entry {count}/{len(ENTRY_IDS)} (ID: {eid})")
            print("time elapsed: ", time.time() - t)
        result = run_single_entry(eid)
        if result is None:
            continue

        aml_mz = result["aml_mz"]
        assignments = result["assignments"]
        parent_formula = result["parent_formula"]

        predicted = 0
        correct = 0
        selected = []

        for a in assignments:
            if a.get("hybrid_score", 0.0) < HYBRID_THR:
                continue

            predicted += 1

            frag_mz = a.get("best_exact_mz")
            matched = is_correct(frag_mz, aml_mz)
            if matched:
                correct += 1

            selected.append((a, matched))

            if RESULTS_OUTPUT.get("enabled") and RESULTS_OUTPUT.get("detail") == "entry_and_fragment":
                fragment_records.append({
                    "schema_version": RESULTS_OUTPUT.get("schema_version", "1.0"),
                    "run_id": run_id,
                    "entry_id": eid,
                    "fragment_rank": predicted,
                    "pred_formula": a.get("best_formula"),
                    "pred_mz": _safe_float(a.get("best_exact_mz")),
                    "score": _safe_float(a.get("score")),
                    "ml_prob": _safe_float(a.get("ml_prob")),
                    "hybrid_score": _safe_float(a.get("hybrid_score")),
                    "is_correct": matched
                })

        # Skip spectra with zero predicted fragments
        if predicted == 0:
            continue

        predicted_counts.append(predicted)
        correct_counts.append(correct)
        precision = correct / predicted
        precisions.append(precision)

        if RESULTS_OUTPUT.get("enabled"):
            score_vals = [_safe_float(a.get("score")) for a, _ in selected]
            ml_vals = [_safe_float(a.get("ml_prob")) for a, _ in selected]
            hybrid_vals = [_safe_float(a.get("hybrid_score")) for a, _ in selected]

            score_mean, score_max, score_min = _score_stats(score_vals)
            ml_mean, _, _ = _score_stats(ml_vals)
            hybrid_mean, _, _ = _score_stats(hybrid_vals)

            entry_records.append({
                "schema_version": RESULTS_OUTPUT.get("schema_version", "1.0"),
                "run_id": run_id,
                "entry_id": eid,
                "parent_formula": parent_formula,
                "precision": precision,
                "predicted_count": predicted,
                "correct_count": correct,
                "score_mean": score_mean,
                "score_max": score_max,
                "score_min": score_min,
                "ml_prob_mean": ml_mean,
                "hybrid_score_mean": hybrid_mean,
                "threshold_hybrid": HYBRID_THR,
                "tolerance_mz": TOL,
                "status": "ok"
            })

        # Export fake HR spectrum
        if RESULTS_OUTPUT.get("include_fake_spectra", True):
            export_fake_spectrum(
                entry_id=eid,
                parent_name=result["entry_id"],
                parent_formula=result["parent_formula"],
                assignments=assignments
            )

    arr = np.array(precisions)
    pred_arr = np.array(predicted_counts)
    corr_arr = np.array(correct_counts)

    print("\n=== Hybrid Precision Statistics (zero-prediction spectra discarded) ===")
    print(f"Spectra evaluated: {len(arr)}")

    print("\n--- Precision (correct / predicted) ---")
    print(f"Mean   : {arr.mean():.4f}")
    print(f"Median : {np.median(arr):.4f}")
    print(f"Stddev : {arr.std():.4f}")
    print(f"Min    : {arr.min():.4f}")
    print(f"Max    : {arr.max():.4f}")
    print(f"25%    : {np.percentile(arr, 25):.4f}")
    print(f"75%    : {np.percentile(arr, 75):.4f}")

    print("\n--- Fragment counts per spectrum ---")
    print(f"Mean predicted fragments : {pred_arr.mean():.2f}")
    print(f"Mean correct fragments   : {corr_arr.mean():.2f}")

    # ------------------------------------------------------------
    # Plot precision distribution
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=15, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title("Hybrid Precision Distribution Across Spectra")
    plt.xlabel("Precision (correct / predicted)")
    plt.ylabel("Number of spectra")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot predicted fragment count distribution
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(pred_arr, bins=15, color="darkorange", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Predicted Fragments per Spectrum")
    plt.xlabel("Number of predicted fragments")
    plt.ylabel("Number of spectra")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------
    # Scatterplot: predicted fragment count vs precision
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_arr, arr, alpha=0.6, color="purple", edgecolor="black")

    plt.title("Precision vs Number of Predicted Fragments")
    plt.xlabel("Number of predicted fragments")
    plt.ylabel("Precision (correct / predicted)")

    # Trend line (optional but useful)
    if len(pred_arr) > 1:
        z = np.polyfit(pred_arr, arr, 1)
        p = np.poly1d(z)
        xs = np.linspace(pred_arr.min(), pred_arr.max(), 200)
        plt.plot(xs, p(xs), "r--", linewidth=2, label="Trend line")
        plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    files_written = {}
    if RESULTS_OUTPUT.get("enabled"):
        run_end = time.time()
        run_summary = {
            "schema_version": RESULTS_OUTPUT.get("schema_version", "1.0"),
            "run_id": run_id,
            "started_at": datetime.fromtimestamp(run_start).isoformat(),
            "finished_at": datetime.fromtimestamp(run_end).isoformat(),
            "config": {
                "hybrid_thr": HYBRID_THR,
                "tol_mz": TOL,
                "detail": RESULTS_OUTPUT.get("detail", "entry")
            },
            "counts": {
                "entries_total": len(ENTRY_IDS),
                "entries_evaluated": len(arr),
                "entries_skipped": len(ENTRY_IDS) - len(arr),
                "fragments_predicted": int(pred_arr.sum()) if len(pred_arr) else 0
            },
            "metrics": {
                "precision_macro_mean": float(arr.mean()) if len(arr) else None,
                "precision_macro_median": float(np.median(arr)) if len(arr) else None
            }
        }

        files_written = save_validation_results(
            config=RESULTS_OUTPUT,
            run_id=run_id,
            entry_records=entry_records,
            fragment_records=fragment_records,
            run_summary=run_summary
        )

    if files_written:
        print("\nSaved validation outputs:")
        for kind, path in files_written.items():
            print(f"- {kind}: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()