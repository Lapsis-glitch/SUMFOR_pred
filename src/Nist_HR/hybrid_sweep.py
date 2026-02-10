# hybrid_sweep.py

import numpy as np
from Large_data import ENTRY_IDS, run_single_entry

TOL = 0.0001

# Sweep ranges
ALPHAS = [0.2, 0.4, 0.6, 0.8]
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.7]
TOP_K_VALUES = [10, 15, 20, 30]


def is_correct(frag_mz, aml_mz_list, tol=TOL):
    if frag_mz is None:
        return False
    return any(abs(frag_mz - m) <= tol for m in aml_mz_list)


def evaluate_config(alpha, thr, top_k):
    precisions = []
    predicted_counts = []
    correct_counts = []

    for eid in ENTRY_IDS[:200]:
        result = run_single_entry(eid)
        if result is None:
            continue

        aml_mz = result["aml_mz"]
        assignments = result["assignments"]

        # Apply hybrid scoring with α
        for a in assignments:
            base = a.get("score", 0.0)
            mlp = a.get("ml_prob", 0.0)
            a["hybrid_score"] = alpha * base + (1 - alpha) * mlp

            # ML fallback
            if base < 0.1 and mlp >= 0.8:
                a["hybrid_score"] = mlp

        # Sort and cap top-K
        assignments.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        assignments = assignments[:top_k]

        predicted = 0
        correct = 0

        for a in assignments:
            if a.get("hybrid_score", 0.0) < thr:
                continue

            predicted += 1
            frag_mz = a.get("best_exact_mz")
            if is_correct(frag_mz, aml_mz):
                correct += 1

        if predicted == 0:
            continue

        predicted_counts.append(predicted)
        correct_counts.append(correct)
        precisions.append(correct / predicted)

    if len(precisions) == 0:
        return None

    arr = np.array(precisions)
    pred_arr = np.array(predicted_counts)
    corr_arr = np.array(correct_counts)

    return {
        "alpha": alpha,
        "threshold": thr,
        "top_k": top_k,
        "mean_precision": arr.mean(),
        "median_precision": np.median(arr),
        "std_precision": arr.std(),
        "mean_predicted": pred_arr.mean(),
        "mean_correct": corr_arr.mean(),
    }


def main():
    results = []

    print("Running hybrid sweep...")

    for alpha in ALPHAS:
        for thr in THRESHOLDS:
            for top_k in TOP_K_VALUES:
                print(f"Testing α={alpha}, thr={thr}, top_k={top_k} ...")
                stats = evaluate_config(alpha, thr, top_k)
                if stats:
                    results.append(stats)

    print("\n=== Sweep Results ===")
    for r in results:
        print(
            f"α={r['alpha']}, thr={r['threshold']}, top_k={r['top_k']}  |  "
            f"mean_precision={r['mean_precision']:.3f}, "
            f"mean_predicted={r['mean_predicted']:.2f}, "
            f"mean_correct={r['mean_correct']:.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()