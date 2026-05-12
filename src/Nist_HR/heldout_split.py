"""
heldout_split.py

Defines the shared held-out test entry list used by both training scripts and
the precision validator.

The held-out entries are picked once (see `generate_heldout_test_set.py`) with
seed=7 — distinct from the random_state=42 used inside both trainers — so the
clean-room test slice is independent of any internal trainer split.

Helpers here intentionally do **not** import Large_data. The trainers operate
on CSV/JSONL artifacts and should not pay the MERGED-JSON load cost just to
read a small ID list.
"""

from __future__ import annotations

import json
from pathlib import Path

HELDOUT_TEST_PATH = Path(__file__).resolve().parent / "heldout_test_entries.json"


def load_heldout_test_ids(path: str | Path = HELDOUT_TEST_PATH) -> set[str]:
    """Return the held-out entry_id set. Missing file → empty set."""
    p = Path(path)
    if not p.exists():
        return set()
    with p.open() as f:
        payload = json.load(f)
    ids = payload.get("entry_ids", []) if isinstance(payload, dict) else payload
    return {str(eid) for eid in ids}


def split_train_pool(all_ids, heldout_ids: set[str] | None = None):
    """Return (train_pool_ids, heldout_test_ids) preserving order of `all_ids`."""
    heldout = heldout_ids if heldout_ids is not None else load_heldout_test_ids()
    train_pool = [eid for eid in all_ids if str(eid) not in heldout]
    test_only = [eid for eid in all_ids if str(eid) in heldout]
    return train_pool, test_only
