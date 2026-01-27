# validate_hybrid_precision_extended.py

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from Large_data import ENTRY_IDS, run_single_entry

TOL = 0.0001
HYBRID_THR = 0.7


def is_correct(frag_mz, aml_mz_list, tol=TOL):
    if frag_mz is None:
        return False
    return any(abs(frag_mz - m) <= tol for m in aml_mz_list)


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
    import os
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

    print("Running hybrid precision validation...")

    for eid in ENTRY_IDS:
        result = run_single_entry(eid)
        if result is None:
            continue

        aml_mz = result["aml_mz"]
        assignments = result["assignments"]
        parent_name = result["entry_id"]
        parent_formula = result["parent_formula"]

        predicted = 0
        correct = 0

        for a in assignments:
            if a.get("hybrid_score", 0.0) < HYBRID_THR:
                continue

            predicted += 1

            frag_mz = a.get("best_exact_mz")
            if is_correct(frag_mz, aml_mz):
                correct += 1

        # Skip spectra with zero predicted fragments
        if predicted == 0:
            continue

        predicted_counts.append(predicted)
        correct_counts.append(correct)
        precisions.append(correct / predicted)

        # Export fake HR spectrum
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

    print("\nDone.")


if __name__ == "__main__":
    main()