"""
tune_scoring_bo.py

Bayesian Optimization for:
- FPS feature weights
- Rule-family priors
- Discriminative scoring parameters

Uses scikit-optimize (skopt) with Gaussian Processes.
"""

import json
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from fps_scoring import FPS_WEIGHTS, RULE_FAMILY_PRIORS
from peak_driven_assignment_engine import SCORING_PARAMS
from runner_peak_driven import run_single_file, COMMON_FILES


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
    - Evaluate average score across training files
    - Return NEGATIVE score (because gp_minimize minimizes)
    """

    # Update FPS weights
    FPS_WEIGHTS["w_cation"]   = params["w_cation"]
    FPS_WEIGHTS["w_hetero"]   = params["w_hetero"]
    FPS_WEIGHTS["w_aromatic"] = params["w_aromatic"]
    FPS_WEIGHTS["w_masspos"]  = params["w_masspos"]
    FPS_WEIGHTS["w_context"]  = params["w_context"]
    FPS_WEIGHTS["w_prior"]    = params["w_prior"]

    # Update discriminative scoring parameters
    SCORING_PARAMS["min_raw_score"]   = params["min_raw_score"]
    SCORING_PARAMS["margin_sharpness"] = params["margin_sharpness"]
    SCORING_PARAMS["depth_penalty"]    = params["depth_penalty"]
    SCORING_PARAMS["intensity_gamma"]  = params["intensity_gamma"]

    # Evaluate on all training files
    scores = []
    for f in COMMON_FILES:
        result = run_single_file(f)
        scores.append(result["score"])

    avg_score = sum(scores) / len(scores)

    print(f"Params: {params}")
    print(f"Avg score: {avg_score:.4f}")

    # gp_minimize MINIMIZES → return negative score
    return -avg_score


# ============================================================
# Run Bayesian optimization
# ============================================================

def main():

    print(f"Training on {len(COMMON_FILES)} files")

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=60,          # number of BO iterations
        n_initial_points=12, # random warmup
        acq_func="EI",       # Expected Improvement
        random_state=42,
    )

    best_params = {
        dim.name: val
        for dim, val in zip(space, result.x)
    }

    print("\n====================================")
    print("Best parameters found:")
    print(json.dumps(best_params, indent=2))
    print("Best score:", -result.fun)
    print("====================================\n")

    with open("best_scoring_params_bo.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("Saved best parameters to best_scoring_params_bo.json")


if __name__ == "__main__":
    main()