"""
plot_fragment_scores.py

Plot is_correct vs. score, ml_prob, and hybrid_score from fragment_metrics.jsonl.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FRAGMENT_METRICS_PATH = "validation_outputs/2026-03-26_16-25-23/fragment_metrics.jsonl"

# --- Load JSONL ---
def load_jsonl(path):
    import json
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

# --- Main plotting function ---
def main(fragment_metrics_path=FRAGMENT_METRICS_PATH):
    df = load_jsonl(fragment_metrics_path)
    # Ensure correct types
    df["is_correct"] = df["is_correct"].astype(bool)
    for col in ["score", "ml_prob", "hybrid_score"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    plot_dir = os.path.join(os.path.dirname(fragment_metrics_path), "fragment_score_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Boxplots: is_correct vs each score/probability
    for col in ["score", "ml_prob", "hybrid_score"]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="is_correct", y=col, data=df)
        plt.title(f"{col} by is_correct")
        plt.xlabel("is_correct")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_by_is_correct_boxplot.png"))
        plt.close()

    # Violin plots
    for col in ["score", "ml_prob", "hybrid_score"]:
        plt.figure(figsize=(6,4))
        sns.violinplot(x="is_correct", y=col, data=df, inner="quartile")
        plt.title(f"{col} by is_correct (violin)")
        plt.xlabel("is_correct")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_by_is_correct_violin.png"))
        plt.close()

    # Swarm plots (if not too many points)
    if len(df) < 5000:
        for col in ["score", "ml_prob", "hybrid_score"]:
            plt.figure(figsize=(6,4))
            sns.swarmplot(x="is_correct", y=col, data=df, alpha=0.5)
            plt.title(f"{col} by is_correct (swarm)")
            plt.xlabel("is_correct")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{col}_by_is_correct_swarm.png"))
            plt.close()

    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
