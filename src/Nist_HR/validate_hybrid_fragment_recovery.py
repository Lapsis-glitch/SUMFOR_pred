"""
validate_hybrid_fragment_recovery.py

Main validation entry-point for the SUMFOR hybrid fragment prediction pipeline.

For every entry in the merged dataset this script:
  1. Runs the full prediction pipeline  (NIST → enumeration → physics → ML → hybrid)
  2. Compares predicted fragment m/z values to high-resolution AML reference peaks
  3. Computes per-entry precision  (correct / predicted)
  4. Prints aggregate statistics and generates matplotlib plots
  5. Optionally saves per-entry and per-fragment metrics to disk

Configuration is controlled by the RESULTS_OUTPUT dict below; set
``"enabled": False`` to suppress all file I/O.

See README.md for the full architecture and usage guide.
"""

# ── Must come before any C-threaded library (numpy, lightgbm, …) ──
# Prevents fork + OpenMP deadlock in worker processes.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
import multiprocessing as mp
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from Large_data import (
    ENTRY_IDS, run_single_entry,
    USE_BEST_PER_PEAK, USE_MULTIPLICATIVE_HYBRID, USE_COMPLEMENTARY_LOSS_FILTER,
    USE_MASS_DEFECT_FILTER, MIN_REL_INTENSITY,
)
from formula import Formula
from chemistry import exact_mass, dbe

# ── Matching thresholds ──────────────────────────────────────
TOL = 0.0003          # Da – mass tolerance for fragment-to-AML matching
HYBRID_THR = 0.80     # minimum hybrid score to count a fragment as "predicted"
MIN_FRAGS_PER_ENTRY = 3  # always keep at least this many top fragments per entry

# ── Parallelism ──────────────────────────────────────────────
N_WORKERS = min(os.cpu_count() or 1, 16)   # leave 2 cores free for OS / tqdm

# ── Output configuration ─────────────────────────────────────
RESULTS_OUTPUT = {
    "enabled": True,                        # master toggle for saving results
    "detail": "entry_and_fragment",         # "entry" | "entry_and_fragment"
    "formats": {
        "entry": "csv",                     # csv | jsonl
        "fragment": "jsonl",                # csv | jsonl
        "run_summary": "json",
    },
    "out_dir": "validation_outputs",        # base directory for run folders
    "run_name": "auto",                     # "auto" → timestamp-based name
    "flush_every": 100,                     # progress message interval
    "include_fake_spectra": False,          # export predicted fragments as JSON
    "schema_version": "1.0",
}


# ── Helper functions ─────────────────────────────────────────

def is_correct(frag_mz, aml_mz_list, tol=TOL):
    """Return True if *frag_mz* is within *tol* Da of any AML reference peak."""
    if frag_mz is None:
        return False
    return any(abs(frag_mz - m) <= tol for m in aml_mz_list)


def _safe_float(value):
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_stats(values):
    """Return (mean, max, min) for a list of nullable floats, or (None, None, None)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    arr = np.array(clean, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min())


# ── Atomic file writers (write-to-tmp then rename) ───────────

def _atomic_write_json(path, data):
    """Write *data* as pretty-printed JSON via a temp file to avoid partial writes."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_jsonl(path, rows):
    """Write *rows* as newline-delimited JSON via a temp file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _atomic_write_csv(path, rows):
    """Write *rows* (list of dicts) as CSV via a temp file."""
    tmp = f"{path}.tmp"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_rows(path_base, fmt, rows):
    """Dispatch to the appropriate atomic writer based on *fmt* ('csv' or 'jsonl')."""
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


# ── Save all validation outputs to a run directory ───────────

def save_validation_results(config, run_id, entry_records, fragment_records, run_summary):
    """
    Persist entry metrics, fragment metrics, config snapshot, and run summary
    into ``<out_dir>/<run_id>/``.

    Returns a dict mapping output kind → file path for everything written.
    """
    if not config.get("enabled", False):
        return {}

    run_dir = os.path.join(config.get("out_dir", "validation_outputs"), run_id)
    os.makedirs(run_dir, exist_ok=True)

    written = {}

    # Entry-level metrics
    entry_format = config.get("formats", {}).get("entry", "csv")
    entry_path = _write_rows(os.path.join(run_dir, "entry_metrics"), entry_format, entry_records)
    if entry_path:
        written["entry"] = entry_path

    # Fragment-level metrics (only when detail == "entry_and_fragment")
    if config.get("detail") == "entry_and_fragment":
        fragment_format = config.get("formats", {}).get("fragment", "jsonl")
        fragment_path = _write_rows(
            os.path.join(run_dir, "fragment_metrics"), fragment_format, fragment_records
        )
        if fragment_path:
            written["fragment"] = fragment_path

    # Config snapshot (always saved)
    config_path = os.path.join(run_dir, "config_snapshot.json")
    _atomic_write_json(config_path, config)
    written["config"] = config_path

    # Run summary
    summary_format = config.get("formats", {}).get("run_summary", "json")
    if summary_format == "json":
        summary_path = os.path.join(run_dir, "run_summary.json")
        _atomic_write_json(summary_path, run_summary)
        written["run_summary"] = summary_path

    return written


# ── Export predicted fragments as a synthetic HR spectrum ─────

def export_fake_spectrum(entry_id, parent_name, parent_formula, assignments,
                         out_dir="fake_spectra"):
    """
    Save high-confidence predicted fragments as a JSON "fake HR spectrum".

    Only fragments with ``hybrid_score >= HYBRID_THR`` are included.
    Each file is written to ``<out_dir>/entry_<entry_id>.json``.
    """
    data = {
        "entry_id": entry_id,
        "parent_name": parent_name,
        "parent_formula": parent_formula,
        "predicted_fragments": [
            {
                "mz": a.get("best_exact_mz"),
                "hybrid_score": a.get("hybrid_score"),
                "formula": a.get("best_formula"),
            }
            for a in assignments
            if a.get("hybrid_score", 0.0) >= HYBRID_THR
        ],
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"entry_{entry_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main validation loop ─────────────────────────────────────

def main():
    """
    Iterate over all dataset entries, run prediction, compute precision,
    print summary statistics, show plots, and save results.
    """
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
    print(f"  Flags: best_per_peak={USE_BEST_PER_PEAK}  "
          f"multiplicative_hybrid={USE_MULTIPLICATIVE_HYBRID}  "
          f"complementary_loss_filter={USE_COMPLEMENTARY_LOSS_FILTER}  "
          f"mass_defect_filter={USE_MASS_DEFECT_FILTER}")
    print(f"  Workers: {N_WORKERS}")

    # ── Parallel pipeline execution ───────────────────────────
    def _init_worker():
        """Silence RDKit's C++ logger inside each forked worker."""
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

    with mp.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(run_single_entry, ENTRY_IDS)

        for result in tqdm(results_iter, total=len(ENTRY_IDS),
                           desc="Validating", unit="entry"):
            if result is None:
                continue

            eid = result["entry_id"]
            aml_mz = result["aml_mz"]
            assignments = result["assignments"]
            parent_formula = result["parent_formula"]

            # ── Select fragments: threshold + minimum guarantee ──
            #
            # Assignments arrive sorted by hybrid_score descending.
            # 1. Keep every fragment above HYBRID_THR.
            # 2. If that gives fewer than MIN_FRAGS_PER_ENTRY, pad
            #    with the next-best fragments (regardless of score)
            #    so every entry contributes at least MIN_FRAGS_PER_ENTRY
            #    predictions.
            above_thr = [a for a in assignments
                         if a.get("hybrid_score", 0.0) >= HYBRID_THR
                         and a.get("best_formula") is not None]

            if len(above_thr) >= MIN_FRAGS_PER_ENTRY:
                selected_assignments = above_thr
            else:
                # Take top-N with a valid formula, regardless of score
                selected_assignments = [a for a in assignments
                                        if a.get("best_formula") is not None
                                        ][:max(MIN_FRAGS_PER_ENTRY, len(above_thr))]

            predicted = 0
            correct = 0
            selected = []  # list of (assignment_dict, is_correct_bool)

            for a in selected_assignments:
                predicted += 1
                frag_mz = a.get("best_exact_mz")
                matched = is_correct(frag_mz, aml_mz)
                if matched:
                    correct += 1
                selected.append((a, matched))

                # Collect per-fragment record
                if (RESULTS_OUTPUT.get("enabled")
                        and RESULTS_OUTPUT.get("detail") == "entry_and_fragment"):

                    frag_mz_val = _safe_float(a.get("best_exact_mz"))
                    nominal = a.get("nominal_mz")

                    # ── Closest AML peak & mass error ──
                    closest_aml = None
                    mass_error_da = None
                    mass_error_ppm = None
                    if frag_mz_val is not None and aml_mz:
                        closest_aml = min(aml_mz, key=lambda m: abs(m - frag_mz_val))
                        mass_error_da = frag_mz_val - closest_aml
                        if closest_aml > 0:
                            mass_error_ppm = (mass_error_da / closest_aml) * 1e6

                    # ── Neutral loss (parent − fragment) ──
                    neutral_loss_da = None
                    if frag_mz_val is not None:
                        try:
                            parent_mass = exact_mass(Formula.from_string(parent_formula))
                            neutral_loss_da = parent_mass - frag_mz_val
                        except Exception:
                            pass

                    # ── Mass defect ──
                    mass_defect = None
                    if frag_mz_val is not None:
                        mass_defect = frag_mz_val - round(frag_mz_val)

                    # ── Fragment DBE ──
                    frag_dbe = None
                    try:
                        frag_f = Formula.from_string(a.get("best_formula", ""))
                        frag_dbe = dbe(frag_f)
                    except Exception:
                        pass

                    # ── Above threshold or kept by minimum guarantee? ──
                    above_threshold = (
                        a.get("hybrid_score", 0.0) >= HYBRID_THR
                    )

                    fragment_records.append({
                        # ── identifiers ──
                        "schema_version": RESULTS_OUTPUT.get("schema_version", "1.0"),
                        "run_id": run_id,
                        "entry_id": eid,
                        "parent_formula": parent_formula,
                        "fragment_rank": predicted,

                        # ── fragment identity ──
                        "pred_formula": a.get("best_formula"),
                        "pred_mz": frag_mz_val,
                        "nominal_mz": nominal,
                        "mass_defect": _safe_float(mass_defect),
                        "frag_dbe": _safe_float(frag_dbe),
                        "neutral_loss_da": _safe_float(neutral_loss_da),

                        # ── generation details ──
                        "rule_source": a.get("rule_source"),
                        "n_candidates_for_peak": len(
                            [x for x in assignments if x.get("nominal_mz") == nominal]
                        ),

                        # ── all scores ──
                        "physics_score": _safe_float(a.get("score")),
                        "ml_prob": _safe_float(a.get("ml_prob")),
                        "hybrid_score": _safe_float(a.get("hybrid_score")),
                        "confidence": _safe_float(a.get("conf")),
                        "peak_intensity": _safe_float(a.get("intensity")),

                        # ── selection info ──
                        "above_hybrid_thr": above_threshold,
                        "kept_by_min_guarantee": not above_threshold,

                        # ── AML matching ──
                        "is_correct": matched,
                        "closest_aml_mz": _safe_float(closest_aml),
                        "mass_error_da": _safe_float(mass_error_da),
                        "mass_error_ppm": _safe_float(mass_error_ppm),
                    })

            # Skip entries with no predicted fragments
            if predicted == 0:
                continue

            precision = correct / predicted
            precisions.append(precision)
            predicted_counts.append(predicted)
            correct_counts.append(correct)

            # Collect per-entry record
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
                    "status": "ok",
                })

            # Optionally export synthetic HR spectrum
            if RESULTS_OUTPUT.get("include_fake_spectra", False):
                export_fake_spectrum(
                    entry_id=eid,
                    parent_name=result["entry_id"],
                    parent_formula=result["parent_formula"],
                    assignments=assignments,
                )

    # ── Aggregate statistics ──────────────────────────────────
    arr = np.array(precisions)
    pred_arr = np.array(predicted_counts)
    corr_arr = np.array(correct_counts)

    print("\n=== Hybrid Precision Statistics (zero-prediction spectra discarded) ===")
    print(f"Spectra evaluated: {len(arr)}")

    print("\n--- Precision (correct / predicted) ---")
    print(f"  Mean   : {arr.mean():.4f}")
    print(f"  Median : {np.median(arr):.4f}")
    print(f"  Stddev : {arr.std():.4f}")
    print(f"  Min    : {arr.min():.4f}")
    print(f"  Max    : {arr.max():.4f}")
    print(f"  25%    : {np.percentile(arr, 25):.4f}")
    print(f"  75%    : {np.percentile(arr, 75):.4f}")

    print("\n--- Fragment counts per spectrum ---")
    print(f"  Mean predicted : {pred_arr.mean():.2f}")
    print(f"  Mean correct   : {corr_arr.mean():.2f}")

    # ── Plots (import matplotlib lazily — after pool is closed) ─

    import matplotlib.pyplot as plt

    # Prepare run directory for saving plots alongside metrics
    plot_dir = None
    if RESULTS_OUTPUT.get("enabled"):
        plot_dir = os.path.join(
            RESULTS_OUTPUT.get("out_dir", "validation_outputs"), run_id
        )
        os.makedirs(plot_dir, exist_ok=True)

    def _save_and_show(fig, filename):
        """Save figure to the run directory (if enabled) and display it."""
        if plot_dir:
            fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
        plt.show()

    # 1. Precision distribution histogram
    fig1 = plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=15, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title("Hybrid Precision Distribution Across Spectra")
    plt.xlabel("Precision (correct / predicted)")
    plt.ylabel("Number of spectra")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig1, "precision_distribution.png")

    # 2. Predicted fragment count distribution
    fig2 = plt.figure(figsize=(8, 5))
    plt.hist(pred_arr, bins=15, color="darkorange", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Predicted Fragments per Spectrum")
    plt.xlabel("Number of predicted fragments")
    plt.ylabel("Number of spectra")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig2, "fragment_count_distribution.png")

    # 3. Precision vs predicted fragment count (scatter + trend)
    fig3 = plt.figure(figsize=(8, 6))
    plt.scatter(pred_arr, arr, alpha=0.6, color="purple", edgecolor="black")
    plt.title("Precision vs Number of Predicted Fragments")
    plt.xlabel("Number of predicted fragments")
    plt.ylabel("Precision (correct / predicted)")
    if len(pred_arr) > 1:
        z = np.polyfit(pred_arr, arr, 1)
        p = np.poly1d(z)
        xs = np.linspace(pred_arr.min(), pred_arr.max(), 200)
        plt.plot(xs, p(xs), "r--", linewidth=2, label="Trend line")
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig3, "precision_vs_fragment_count.png")

    # ── Save results to disk ──────────────────────────────────
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
                "min_frags_per_entry": MIN_FRAGS_PER_ENTRY,
                "tol_mz": TOL,
                "min_rel_intensity": MIN_REL_INTENSITY,
                "detail": RESULTS_OUTPUT.get("detail", "entry"),
                "use_best_per_peak": USE_BEST_PER_PEAK,
                "use_multiplicative_hybrid": USE_MULTIPLICATIVE_HYBRID,
                "use_complementary_loss_filter": USE_COMPLEMENTARY_LOSS_FILTER,
                "use_mass_defect_filter": USE_MASS_DEFECT_FILTER,
            },
            "counts": {
                "entries_total": len(ENTRY_IDS),
                "entries_evaluated": len(arr),
                "entries_skipped": len(ENTRY_IDS) - len(arr),
                "fragments_predicted": int(pred_arr.sum()) if len(pred_arr) else 0,
            },
            "metrics": {
                "precision_macro_mean": float(arr.mean()) if len(arr) else None,
                "precision_macro_median": float(np.median(arr)) if len(arr) else None,
            },
        }

        files_written = save_validation_results(
            config=RESULTS_OUTPUT,
            run_id=run_id,
            entry_records=entry_records,
            fragment_records=fragment_records,
            run_summary=run_summary,
        )

    if files_written:
        print("\nSaved validation outputs:")
        for kind, path in files_written.items():
            print(f"  {kind}: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

