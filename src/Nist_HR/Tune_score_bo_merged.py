"""
tune_scoring_bo_merged.py

Bayesian Optimization over merged_clean.json with a train/validation split.
Optimizes:
- FPS feature weights
- discriminative scoring parameters

Uses scikit-optimize (skopt) with Gaussian Processes.
"""

import json
import random
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from fps_scoring import FPS_WEIGHTS
from peak_driven_assignment_engine import SCORING_PARAMS
from Large_data import run_single_entry, ENTRY_IDS   # uses cleaned dataset


# ============================================================
# Train/Validation split
# ============================================================

random.seed(42)

ENTRY_IDS_INT = [int(e) for e in ENTRY_IDS]
ENTRY_IDS_INT.sort()

# 80% train, 20% validation
split_idx = int(0.8 * len(ENTRY_IDS_INT))
TRAIN_IDS = ENTRY_IDS_INT[:split_idx]
VAL_IDS   = ENTRY_IDS_INT[split_idx:]

print(f"Training on {len(TRAIN_IDS)} entries")
print(f"Validating on {len(VAL_IDS)} entries")


# ============================================================
# Define search space
# ============================================================

space = [

    # FPS feature weights
    Real(0.05, 0.6, name="w_cation"),
    Real(0.05, 0.6, name="w_hetero"),
    Real(0.05, 0.6, name="w_aromatic"),
    Real(0.05, 0.6, name="w_masspos"),
    Real(0.05, 0.6, name="w_context"),
    Real(0.05, 0.6, name="w_prior"),

    # Discriminative scoring parameters
    Real(0.01, 0.15, name="min_raw_score"),
    Real(2.0, 15.0, name="margin_sharpness"),
    Real(0.05, 0.6, name="depth_penalty"),
    Real(0.5, 3.0, name="intensity_gamma"),
]


# ============================================================
# Objective function for Bayesian optimization
# ============================================================

@use_named_args(space)
def objective(**params):
    """
    Bayesian optimization objective:
    - Update FPS_WEIGHTS
    - Update SCORING_PARAMS
    - Evaluate average score on TRAIN set
    - Return NEGATIVE score (gp_minimize minimizes)
    """

    # Update FPS weights
    FPS_WEIGHTS["w_cation"]   = params["w_cation"]
    FPS_WEIGHTS["w_hetero"]   = params["w_hetero"]
    FPS_WEIGHTS["w_aromatic"] = params["w_aromatic"]
    FPS_WEIGHTS["w_masspos"]  = params["w_masspos"]
    FPS_WEIGHTS["w_context"]  = params["w_context"]
    FPS_WEIGHTS["w_prior"]    = params["w_prior"]

    # Update discriminative scoring parameters
    SCORING_PARAMS["min_raw_score"]    = params["min_raw_score"]
    SCORING_PARAMS["margin_sharpness"] = params["margin_sharpness"]
    SCORING_PARAMS["depth_penalty"]    = params["depth_penalty"]
    SCORING_PARAMS["intensity_gamma"]  = params["intensity_gamma"]

    # Evaluate on training set
    scores = []
    for eid in TRAIN_IDS:
        result = run_single_entry(str(eid))
        scores.append(result["score"])

    avg_score = sum(scores) / len(scores)

    print(f"Params: {params}")
    print(f"Train avg score: {avg_score:.4f}")

    return -avg_score   # gp_minimize minimizes


# ============================================================
# Run Bayesian optimization
# ============================================================

def main():

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=60,          # total BO iterations
        n_initial_points=12, # random warmup
        acq_func="EI",       # Expected Improvement
        random_state=42,
    )

    best_params = {
        dim.name: val
        for dim, val in zip(space, result.x)
    }

    print("\n====================================")
    print("Best parameters found (train set):")
    print(json.dumps(best_params, indent=2))
    print("Train score:", -result.fun)
    print("====================================\n")

    # --------------------------------------------------------
    # Evaluate on validation set
    # --------------------------------------------------------
    print("Evaluating on validation set...\n")

    # Apply best params
    for key in FPS_WEIGHTS:
        if key in best_params:
            FPS_WEIGHTS[key] = best_params[key]

    for key in SCORING_PARAMS:
        if key in best_params:
            SCORING_PARAMS[key] = best_params[key]

    val_scores = []
    for eid in VAL_IDS:
        result_val = run_single_entry(str(eid))
        val_scores.append(result_val["score"])
        print(f"Entry {eid}: score={result_val['score']:.4f}")

    val_avg = sum(val_scores) / len(val_scores)
    print("\n====================================")
    print("Validation average score:", val_avg)
    print("====================================")

    # Save best parameters
    with open("best_scoring_params_bo_merged.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("Saved best parameters to best_scoring_params_bo_merged.json")


if __name__ == "__main__":
    main()