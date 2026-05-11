"""
plot_fragment_scores.py

Visualise the relationship between fragment correctness and the three
scoring channels (physics ``score``, ``ml_prob``, ``hybrid_score``).

Reads ``fragment_metrics.jsonl`` produced by a validation run and generates:
  - Box-plots    — score distributions split by ``is_correct``
  - Violin plots — same data, showing density shape
  - Swarm plots  — individual data points (only when n < 5 000)

Usage
-----
CLI::

    python plot_fragment_scores.py <path/to/fragment_metrics.jsonl>

IDE (PyCharm)::

    Run directly — uses the default FRAGMENT_METRICS_PATH constant.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Configuration ────────────────────────────────────────────
FRAGMENT_METRICS_PATH = "validation_outputs/2026-03-26_16-25-23/fragment_metrics.jsonl"
SCORE_COLUMNS = ["score", "ml_prob", "hybrid_score"]
SWARM_MAX_ROWS = 5000  # skip swarm plots above this row count


# ── JSONL loader ─────────────────────────────────────────────

def load_jsonl(path):
    """Read a newline-delimited JSON file into a DataFrame, skipping bad lines."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


# ── Plotting helpers ─────────────────────────────────────────

def _save_plot(fig, plot_dir, filename):
    """Save figure and close."""
    fig.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)


# ── Main plotting function ───────────────────────────────────

def main(fragment_metrics_path=FRAGMENT_METRICS_PATH):
    """Load fragment metrics and generate box / violin / swarm plots."""
    df = load_jsonl(fragment_metrics_path)

    # Ensure correct column types
    df["is_correct"] = df["is_correct"].astype(bool)
    for col in SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_dir = os.path.join(os.path.dirname(fragment_metrics_path), "fragment_score_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Box-plots ────────────────────────────────────────────
    for col in SCORE_COLUMNS:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="is_correct", y=col, data=df, ax=ax)
        ax.set_title(f"{col} by is_correct")
        fig.tight_layout()
        _save_plot(fig, plot_dir, f"{col}_by_is_correct_boxplot.png")

    # ── Violin plots ─────────────────────────────────────────
    for col in SCORE_COLUMNS:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(x="is_correct", y=col, data=df, inner="quartile", ax=ax)
        ax.set_title(f"{col} by is_correct (violin)")
        fig.tight_layout()
        _save_plot(fig, plot_dir, f"{col}_by_is_correct_violin.png")

    # ── Swarm plots (skip if dataset is large) ───────────────
    if len(df) < SWARM_MAX_ROWS:
        for col in SCORE_COLUMNS:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.swarmplot(x="is_correct", y=col, data=df, alpha=0.5, ax=ax)
            ax.set_title(f"{col} by is_correct (swarm)")
            fig.tight_layout()
            _save_plot(fig, plot_dir, f"{col}_by_is_correct_swarm.png")

    print(f"Plots saved to {plot_dir}")


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
