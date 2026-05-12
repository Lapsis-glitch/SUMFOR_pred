# validate_hybrid_model.py

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)

from Large_data import ENTRY_IDS, run_single_entry


TOL = 0.0001  # AML m/z match tolerance (HR, so very tight)


def label_assignment_correct(frag_mz, aml_mz_list, tol=TOL):
    if frag_mz is None:
        return 0
    return 1 if any(abs(frag_mz - m) <= tol for m in aml_mz_list) else 0


def main():
    y_true = []
    physics_scores = []
    ml_scores = []
    hybrid_scores = []

    n_entries = 0

    print("Running validation over merged dataset...")

    for eid in ENTRY_IDS[:50]:
        result = run_single_entry(eid)
        if result is None:
            continue

        n_entries += 1

        aml_mz = result["aml_mz"]
        assignments = result["assignments"]

        for a in assignments:
            frag_mz = a.get("best_exact_mz")

            label = label_assignment_correct(frag_mz, aml_mz)
            y_true.append(label)

            physics_scores.append(a.get("score", 0.0))
            ml_scores.append(a.get("ml_prob", 0.0))
            hybrid_scores.append(a.get("hybrid_score", 0.0))

    print(f"\nUsed {n_entries} entries for validation.")
    print(f"Total assignments: {len(y_true)}")

    # Safety check
    if len(set(y_true)) < 2:
        print("Not enough positive/negative labels for meaningful metrics.")
        return

    def binarize(scores, thr=0.5):
        return [1 if s >= thr else 0 for s in scores]

    # Physics-only
    print("\n=== Physics-only scoring (score) ===")
    print("AUC:", roc_auc_score(y_true, physics_scores))
    print("AP :", average_precision_score(y_true, physics_scores))
    print("ACC:", accuracy_score(y_true, binarize(physics_scores)))
    print("F1 :", f1_score(y_true, binarize(physics_scores)))

    # ML-only
    print("\n=== ML-only scoring (ml_prob) ===")
    print("AUC:", roc_auc_score(y_true, ml_scores))
    print("AP :", average_precision_score(y_true, ml_scores))
    print("ACC:", accuracy_score(y_true, binarize(ml_scores)))
    print("F1 :", f1_score(y_true, binarize(ml_scores)))

    # Hybrid
    print("\n=== Hybrid scoring (score × ml_prob) ===")
    print("AUC:", roc_auc_score(y_true, hybrid_scores))
    print("AP :", average_precision_score(y_true, hybrid_scores))
    print("ACC:", accuracy_score(y_true, binarize(hybrid_scores)))
    print("F1 :", f1_score(y_true, binarize(hybrid_scores)))


if __name__ == "__main__":
    main()