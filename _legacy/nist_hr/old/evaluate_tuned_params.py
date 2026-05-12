"""
evaluate_tuned_params.py

Loads tuned parameters (from Bayesian optimization or random search),
applies them to the scoring engine, and evaluates performance across
all NIST/AML pairs using run_single_file().
"""

import json
from fps_scoring import FPS_WEIGHTS, RULE_FAMILY_PRIORS
from peak_driven_assignment_engine import SCORING_PARAMS
from runner_peak_driven import run_single_file, COMMON_FILES


# ------------------------------------------------------------
# Load tuned parameters
# ------------------------------------------------------------

PARAM_FILE = "best_scoring_params_bo.json"   # or best_scoring_params.json

with open(PARAM_FILE, "r") as f:
    tuned = json.load(f)

print("Loaded tuned parameters:")
print(json.dumps(tuned, indent=2))


# ------------------------------------------------------------
# Apply tuned parameters
# ------------------------------------------------------------

# FPS feature weights
for key in FPS_WEIGHTS:
    if key in tuned:
        FPS_WEIGHTS[key] = tuned[key]

# Rule-family priors (optional: only update if present)
for key in RULE_FAMILY_PRIORS:
    if key in tuned:
        RULE_FAMILY_PRIORS[key] = tuned[key]

# Discriminative scoring parameters
for key in SCORING_PARAMS:
    if key in tuned:
        SCORING_PARAMS[key] = tuned[key]


print("\nApplied tuned parameters to scoring engine.\n")


# ------------------------------------------------------------
# Final evaluation
# ------------------------------------------------------------

scores = []
details = []

print(f"Evaluating tuned model on {len(COMMON_FILES)} files...\n")

for file in COMMON_FILES:
    result = run_single_file(file)
    scores.append(result["score"])
    details.append((file, result["score"], result["n_matches"], result["n_total"]))

    print(f"{file}: score={result['score']:.4f}  "
          f"({result['n_matches']}/{result['n_total']} matches)")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

avg = sum(scores) / len(scores) if scores else 0.0

print("\n====================================")
print(" FINAL EVALUATION SUMMARY")
print("====================================")
print(f"Average score: {avg:.4f}")
print(f"Best file:  {max(details, key=lambda x: x[1])[0]}")
print(f"Worst file: {min(details, key=lambda x: x[1])[0]}")
print("====================================\n")