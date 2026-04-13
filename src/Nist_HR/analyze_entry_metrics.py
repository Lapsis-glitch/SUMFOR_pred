"""
analyze_entry_metrics.py

Analyze entry_metrics.csv/jsonl and plot precision vs. chemical descriptors.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from formula import Formula
from chemistry import dbe, exact_mass
from chemical_classification import classify_from_formula
import json

# Path to merged dataset (update if needed)
MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"

# --- Utility: parse formula and extract descriptors ---
def extract_descriptors(formula_str):
    f = Formula.from_string(formula_str)
    elems = f.elements
    desc = {}
    desc["total_atoms"] = sum(elems.values())
    desc["num_C"] = elems.get("C", 0)
    desc["num_H"] = elems.get("H", 0)
    desc["num_N"] = elems.get("N", 0)
    desc["num_O"] = elems.get("O", 0)
    desc["num_S"] = elems.get("S", 0)
    desc["num_P"] = elems.get("P", 0)
    desc["num_F"] = elems.get("F", 0)
    desc["num_Cl"] = elems.get("Cl", 0)
    desc["num_Br"] = elems.get("Br", 0)
    desc["num_I"] = elems.get("I", 0)
    desc["num_halogen"] = sum([desc["num_F"], desc["num_Cl"], desc["num_Br"], desc["num_I"]])
    desc["dbe"] = dbe(f)
    desc["exact_mass"] = exact_mass(f, charged=False)
    # Ratios (avoid div by zero)
    desc["H_C"] = desc["num_H"] / desc["num_C"] if desc["num_C"] else None
    desc["O_C"] = desc["num_O"] / desc["num_C"] if desc["num_C"] else None
    desc["N_C"] = desc["num_N"] / desc["num_C"] if desc["num_C"] else None
    desc["S_C"] = desc["num_S"] / desc["num_C"] if desc["num_C"] else None
    desc["Hal_C"] = desc["num_halogen"] / desc["num_C"] if desc["num_C"] else None
    # Ring/aromaticity: use classification
    classes = classify_from_formula(f)
    desc["has_ring"] = int("aromatic" in classes or "cyclic" in classes or "heterocycle" in classes)
    desc["is_aromatic"] = int("aromatic" in classes)
    desc["is_heterocycle"] = int("heterocycle" in classes)
    desc["is_halogenated"] = int("halogenated" in classes)
    desc["is_nitrogenous"] = int("nitrogenous" in classes)
    desc["is_oxygenated"] = int("oxygenated" in classes)
    desc["is_sulfur"] = int("sulfur" in classes)
    return desc

# --- Utility: extract BDE and spectrum stats from merged JSON ---
def extract_merged_metrics(entry_id, merged):
    d = merged.get(str(entry_id), {})
    out = {}
    # BDE data
    bde = d.get("bde_data", {})
    if not isinstance(bde, dict):
        bde = {}
    bonds = bde.get("bonds", [])
    if bonds is None:
        bonds = []
    bde_vals = [b["bde"] for b in bonds if b and b.get("bde") is not None]
    out["bde_min"] = min(bde_vals) if bde_vals else None
    out["bde_max"] = max(bde_vals) if bde_vals else None
    out["bde_mean"] = sum(bde_vals)/len(bde_vals) if bde_vals else None
    out["bde_median"] = sorted(bde_vals)[len(bde_vals)//2] if bde_vals else None
    out["bde_count"] = len(bde_vals)
    out["bde_lt80"] = sum(1 for v in bde_vals if v < 80)
    out["bde_lt100"] = sum(1 for v in bde_vals if v < 100)
    out["bde_lt120"] = sum(1 for v in bde_vals if v < 120)
    out["num_atoms_bde"] = bde.get("num_atoms")
    out["num_bonds_bde"] = bde.get("num_bonds")
    # Bond type counts
    for t in ["C-C", "C-H", "C-O", "C-N", "C-Cl", "C-Br", "C-F", "C-I", "O-H", "N-H"]:
        out[f"bonds_{t}"] = sum(
            1 for b in bonds if b and b.get("bde") is not None and
            ((b.get("atom1_symbol")+"-"+b.get("atom2_symbol") == t) or (b.get("atom2_symbol")+"-"+b.get("atom1_symbol") == t))
        )
    # NIST/AML spectrum stats
    nist = d.get("nist_matches", [{}])[0] if d.get("nist_matches") else {}
    nist_mz = nist.get("mz", []) or []
    nist_int = nist.get("intensities", []) or []
    out["nist_peak_count"] = len(nist_mz)
    out["nist_int_max"] = max(nist_int) if nist_int else None
    out["nist_int_mean"] = sum(nist_int)/len(nist_int) if nist_int else None
    out["nist_int_median"] = sorted(nist_int)[len(nist_int)//2] if nist_int else None
    csv_entry = d.get("csv_entry", {})
    aml_mz = csv_entry.get("mz", []) or []
    aml_int = csv_entry.get("intensities", []) or []
    out["aml_peak_count"] = len(aml_mz)
    out["aml_int_max"] = max(aml_int) if aml_int else None
    out["aml_int_mean"] = sum(aml_int)/len(aml_int) if aml_int else None
    out["aml_int_median"] = sorted(aml_int)[len(aml_int)//2] if aml_int else None
    # Retention index, match score
    out["ri"] = nist.get("estimated_kovats_ri")
    out["match_score"] = nist.get("match_score")
    return out

# --- Main analysis function ---
def main(entry_metrics_path):
    # Auto-detect format
    ext = os.path.splitext(entry_metrics_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(entry_metrics_path)
    elif ext == ".jsonl":
        df = pd.read_json(entry_metrics_path, lines=True)
    else:
        raise ValueError("Unsupported file format: " + ext)

    # Load merged JSON for extra metrics
    with open(MERGED_PATH, "r") as f:
        merged = json.load(f)

    # Extract descriptors
    descs = df["parent_formula"].apply(extract_descriptors)
    desc_df = pd.DataFrame(list(descs))
    # Extract merged metrics
    merged_metrics = df["entry_id"].apply(lambda eid: extract_merged_metrics(eid, merged))
    merged_df = pd.DataFrame(list(merged_metrics))
    df = pd.concat([df, desc_df, merged_df], axis=1)

    # Plot: precision vs. each descriptor
    plot_dir = os.path.join(os.path.dirname(entry_metrics_path), "analysis_plots")
    os.makedirs(plot_dir, exist_ok=True)
    descriptors = [
        "total_atoms", "num_C", "num_H", "num_N", "num_O", "num_S", "num_P",
        "num_halogen", "dbe", "exact_mass", "has_ring", "is_aromatic", "is_heterocycle",
        "is_halogenated", "is_nitrogenous", "is_oxygenated", "is_sulfur",
        "H_C", "O_C", "N_C", "S_C", "Hal_C",
        "bde_min", "bde_max", "bde_mean", "bde_median", "bde_count", "bde_lt80", "bde_lt100", "bde_lt120",
        "num_atoms_bde", "num_bonds_bde",
        "nist_peak_count", "nist_int_max", "nist_int_mean", "nist_int_median",
        "aml_peak_count", "aml_int_max", "aml_int_mean", "aml_int_median",
        "ri", "match_score"
    ] + [f"bonds_{t}" for t in ["C-C", "C-H", "C-O", "C-N", "C-Cl", "C-Br", "C-F", "C-I", "O-H", "N-H"]]
    for desc in descriptors:
        if desc not in df.columns:
            continue
        plt.figure(figsize=(6,4))
        if df[desc].nunique() < 10:
            sns.boxplot(x=desc, y="precision", data=df)
        else:
            sns.scatterplot(x=desc, y="precision", data=df, alpha=0.5)
        plt.title(f"Precision vs. {desc}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"precision_vs_{desc}.png"))
        plt.close()

    # Coerce all descriptor columns to numeric for correlation
    for desc in descriptors:
        if desc in df.columns:
            df[desc] = pd.to_numeric(df[desc], errors='coerce')
    df["precision"] = pd.to_numeric(df["precision"], errors='coerce')

    # Only use numeric columns for correlation
    corr_cols = ["precision"] + [d for d in descriptors if d in df.columns and pd.api.types.is_numeric_dtype(df[d])]
    dropped = [d for d in descriptors if d in df.columns and not pd.api.types.is_numeric_dtype(df[d])]
    if dropped:
        print(f"[WARN] Dropping non-numeric columns from correlation: {dropped}")

    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix: precision and descriptors")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()

    print(f"Analysis complete. Plots saved to {plot_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        # Test mode: process first 5 entries and print
        test_path = os.path.join(os.path.dirname(__file__), "validation_outputs", "2026-03-26_16-25-23", "entry_metrics.csv")
        print(f"[TEST MODE] Using: {test_path}")
        df = pd.read_csv(test_path)
        with open(MERGED_PATH, "r") as f:
            merged = json.load(f)
        for i, row in df.head(5).iterrows():
            print(f"Entry {row['entry_id']} ({row['parent_formula']}):")
            print("  Descriptors:", extract_descriptors(row['parent_formula']))
            print("  Merged metrics:", extract_merged_metrics(row['entry_id'], merged))
        print("[TEST MODE DONE]")
    elif len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        # IDE/interactive mode: prompt for file or use default
        print("No input file provided as argument.")
        default_path = os.path.join(os.path.dirname(__file__), "validation_outputs", "2026-03-26_16-25-23", "entry_metrics.csv")
        print(f"Enter path to entry_metrics.csv or .jsonl [default: {default_path}]: ", end="")
        user_path = input().strip()
        if not user_path:
            user_path = default_path
        if not os.path.isfile(user_path):
            print(f"File not found: {user_path}")
            exit(1)
        main(user_path)
