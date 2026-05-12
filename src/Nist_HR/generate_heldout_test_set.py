"""
generate_heldout_test_set.py

Produce src/Nist_HR/heldout_test_entries.json — the shared 15% test slice that
neither train_ml_model_v2.py nor train_final_decision_calibrator.py should
ever train on, and that validate_hybrid_fragment_recovery_precision.py can be
restricted to via SUMFOR_RUN_MODE=test_only.

Seed is 7 (deliberately distinct from the 42 used inside both trainers) so
the holdout is independent of any internal trainer split.

Usage:
    cd src/Nist_HR
    python generate_heldout_test_set.py
    python generate_heldout_test_set.py --test-size 0.10 --seed 13 --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from heldout_split import HELDOUT_TEST_PATH

MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick a held-out test entry list once.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of entries to hold out.")
    parser.add_argument("--seed", type=int, default=7, help="random_state for the split.")
    parser.add_argument("--merged-path", default=MERGED_PATH, help="Path to the merged dataset JSON.")
    parser.add_argument("--output", default=str(HELDOUT_TEST_PATH), help="Where to write the holdout list.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing {output_path}. Pass --force if you mean to regenerate."
        )

    with open(args.merged_path) as f:
        merged = json.load(f)
    all_ids = sorted(merged.keys(), key=lambda x: int(x))

    _, holdout_ids = train_test_split(
        all_ids, test_size=args.test_size, random_state=args.seed
    )
    holdout_ids = sorted(holdout_ids, key=lambda x: int(x))

    payload = {
        "entry_ids": holdout_ids,
        "metadata": {
            "n_total_entries": len(all_ids),
            "n_heldout": len(holdout_ids),
            "test_size": args.test_size,
            "seed": args.seed,
            "merged_path": args.merged_path,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {output_path}")
    print(f"  total entries : {len(all_ids)}")
    print(f"  held out      : {len(holdout_ids)} ({100 * len(holdout_ids) / len(all_ids):.2f}%)")
    print(f"  seed          : {args.seed}")


if __name__ == "__main__":
    main()
