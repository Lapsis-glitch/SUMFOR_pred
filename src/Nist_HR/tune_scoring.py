"""
tune_scoring.py

Parameter tuner for:
- FPS feature weights
- Rule-family priors
- Discriminative scoring parameters
- Depth penalty
- Intensity exponent

Uses random search over parameter space and evaluates
performance using AML/NIST match scores.
"""

import random
import json
import time
from pathlib import Path

# Import tunable globals
from fps_scoring import FPS_WEIGHTS, RULE_FAMILY_PRIORS
from peak_driven_assignment_engine import SCORING_PARAMS
from runner_peak_driven import run_single_file


# ============================================================
# Parameter search space
# ============================================================

FEATURE_WEIGHT_SPACE = {
    "w_cation":   (0.05, 0.6),
    "w_hetero":   (0.05, 0.6),
    "w_aromatic": (0.05, 0.6),
    "w_masspos":  (0.05, 0.6),
    "w_context":  (0.05, 0.6),
    "w_prior":    (0.05, 0.6),
}

RULE_PRIOR_SPACE = {
    key: (0.2, 1.2)
    for key in RULE_FAMILY_PRIORS.keys()
}

SCORING_PARAM_SPACE = {
    "min_raw_score":   (0.01, 0.15),
    "margin_sharpness": (2.0, 15.0),
    "depth_penalty":    (0.05, 0.6),
    "intensity_gamma":  (0.5, 3.0),
}


# ============================================================
# Random sampling helpers
# ============================================================

def sample_from_range(rng):
    lo, hi = rng
    return lo + random.random() * (hi - lo)


def sample_feature_weights():
    return {
        key: sample_from_range(rng)
        for key, rng in FEATURE_WEIGHT_SPACE.items()
    }


def sample_rule_priors():
    return {
        key: sample_from_range(rng)
        for key, rng in RULE_PRIOR_SPACE.items()
    }


def sample_scoring_params():
    return {
        key: sample_from_range(rng)
        for key, rng in SCORING_PARAM_SPACE.items()
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_params(fps_w, rule_p, scoring_p, files):
    """
    Inject parameters, run pipeline on each file, return average score.
    """

    # Inject FPS weights
    FPS_WEIGHTS.update(fps_w)

    # Inject rule-family priors
    RULE_FAMILY_PRIORS.update(rule_p)

    # Inject discriminative scoring parameters
    SCORING_PARAMS.update(scoring_p)

    scores = []
    for f in files:
        result = run_single_file(f)
        scores.append(result["score"])

    return sum(scores) / len(scores)


# ============================================================
# Main tuning loop
# ============================================================

def main():
    # Training set: all matching NIST/AML files
    # You can also manually specify a subset here
    training_dir = Path("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_Ref/")
    files = sorted([f.name for f in training_dir.glob("*.jdx")])

    N_ITER = 120  # number of random samples
    best_score = -1
    best_params = None

    print(f"Tuning over {N_ITER} random samples...\n")

    for i in range(N_ITER):
        fps_w = sample_feature_weights()
        rule_p = sample_rule_priors()
        scoring_p = sample_scoring_params()

        score = evaluate_params(fps_w, rule_p, scoring_p, files)

        print(f"[{i+1}/{N_ITER}] Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = {
                "fps_weights": fps_w,
                "rule_priors": rule_p,
                "scoring_params": scoring_p,
            }

            print("  New best!")
            print(json.dumps(best_params, indent=2))

    print("\n====================================")
    print("Best score:", best_score)
    print("Best parameters:")
    print(json.dumps(best_params, indent=2))

    # Save to disk
    with open("best_scoring_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nSaved best parameters to best_scoring_params.json")


if __name__ == "__main__":
    main()