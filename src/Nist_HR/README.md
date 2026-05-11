# SUMFOR Hybrid Fragment Validation Pipeline

## Overview

This pipeline predicts EI mass-spectrum fragment formulas from a **low-resolution NIST spectrum** and validates them against **high-resolution AML reference peaks**. It combines physics-based fragmentation rules with BDE-driven structural fragmentation and a LightGBM ML correction model to produce a **hybrid confidence score** for each predicted fragment.

The central validation script is **`validate_hybrid_fragment_recovery.py`**. It iterates over every entry in a merged JSON dataset, runs the full prediction pipeline, and measures **precision** (fraction of predicted fragments that match a real AML peak within a mass tolerance).

---

## Directory Structure

```
Nist_HR/
├── validate_hybrid_fragment_recovery.py   ← main entry point
├── Large_data.py                          ← data loader & pipeline runner
│
├── formula.py                             ← Formula dataclass
├── chemistry.py                           ← exact_mass(), dbe()
├── chemical_classification.py             ← compound class tagger
│
├── peak_driven_enumerator.py              ← rule-based fragment enumerator
├── recursive_fragmenter.py                ← recursive rule application
├── fragmentation_rules.py                 ← rule orchestrator
├── fragmentation_rules_info/              ← modular rule packs
│   ├── base.py, registry.py
│   ├── universal.py, alcohol.py, alkene.py, amine.py, aromatic.py
│   ├── carbonyl.py, ester.py, ether.py, halogen.py
│   ├── phosphorus.py, sulfur.py
│   └── __init__.py
├── hybrid_enumerator.py                   ← rules + BDE enumerator
├── MLFF_fragmentation/                    ← BDE-driven fragmentation
│   ├── bde_enumerator.py
│   ├── bde_fragmenter.py
│   └── __init__.py
│
├── peak_driven_assignment_engine.py       ← assigns fragments to NIST peaks
├── assignment_engine.py                   ← legacy engine (used internally)
├── peakassignment.py                      ← PeakAssignment dataclass
├── fragment.py                            ← Fragment dataclass
├── fps_scoring.py                         ← Fragment Plausibility Score
├── utils.py                               ← convert_assignments(), CSV helpers
│
├── ml_correction.py                       ← LightGBM model class
├── ml_correction_integration.py           ← ML integration wrapper
├── ml_correction_model_SIP.txt            ← trained model weights
├── ml_feature_importance_SIP.json         ← feature names
│
├── analyze_entry_metrics.py               ← post-hoc entry analysis
├── plot_fragment_scores.py                ← post-hoc fragment plots
├── __init__.py
├── README.md
├── validation_outputs/                    ← auto-generated run results
│
└── old/                                   ← deprecated / offline-only files
    ├── runner.py, runner_peak_driven.py
    ├── enumerator.py, scoring.py, JDXParser.py
    ├── train_ml_model.py, train_ml_model_Kfold.py
    ├── collect_training_data.py, collect_training_data_bde.py
    ├── tune_scoring.py, tune_scoring_bo.py, Tune_score_bo_merged.py
    ├── evaluate_tuned_params.py, hybrid_sweep.py, hybrid_s.py
    ├── validate_ml_correction.py, clean_dataset.py
    ├── fragmentation_rules_old.py
    ├── training_fragments*.csv, best_scoring_params*.json
    ├── ml_correction_model.txt, ml_feature_importance.json
    ├── fake_spectra*/ 
    └── MLFF_fragmentation/  (dev scripts)
```

---

## Architecture at a Glance

```
merged_clean_SIP_with_pubchem_bde.json   (ground-truth dataset)
              │
              ▼
        Large_data.py                    (data loader + full pipeline runner)
              │
     ┌────────┴────────────────────────┐
     │                                 │
     ▼                                 ▼
PeakDrivenEnumerator             HybridEnumerator
  (rule-based only)             (rules + BDE structure)
     │                                 │
     │  ┌──────────────────────────────┘
     │  │
     ▼  ▼
PeakDrivenAssignmentEngine        (assigns fragments to NIST peaks)
              │
              ▼
   ml_correction_integration.py   (LightGBM re-scoring)
              │
              ▼
        hybrid_score              (α·physics + (1-α)·ML)
              │
              ▼
validate_hybrid_fragment_recovery.py   ← YOU ARE HERE
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
 entry_metrics     fragment_metrics     (saved results)
     │                 │
     ▼                 ▼
analyze_entry_metrics.py    plot_fragment_scores.py   (post-hoc analysis)
```

---

## File Reference

### Core Pipeline

| File | Purpose |
|------|---------|
| **`validate_hybrid_fragment_recovery.py`** | Main validation entry point. Runs every dataset entry through the prediction pipeline, compares predicted fragment m/z values to AML reference peaks, computes precision, prints statistics, generates plots, and (optionally) saves per-entry and per-fragment metrics to disk. |
| **`Large_data.py`** | Data loader and single-entry pipeline runner. Loads the merged JSON dataset at import time, exposes `ENTRY_IDS` and `run_single_entry(entry_id)` which executes the full NIST → enumeration → physics → ML → hybrid pipeline for one compound. |
| **`formula.py`** | Lightweight, immutable `Formula` dataclass. Parses formula strings (`"C6H6"`) into element-count dicts. Used everywhere. |
| **`chemistry.py`** | Chemical utility functions: monoisotopic `exact_mass()`, double-bond equivalents `dbe()`. Uses explicit IUPAC monoisotopic masses. |
| **`chemical_classification.py`** | Rule-based chemical class tagger. Classifies a compound as aromatic, halogenated, nitrogenous, etc., from its formula and/or name. |

### Fragment Enumeration

| File | Purpose |
|------|---------|
| **`peak_driven_enumerator.py`** | Rule-based fragment enumerator. Pre-computes all rule-derivable sub-formulas up to `max_depth` and indexes them by nominal mass for fast lookup. |
| **`recursive_fragmenter.py`** | Recursively applies fragmentation rules to generate primary, secondary, and deeper fragments. Called internally by `PeakDrivenEnumerator`. |
| **`fragmentation_rules.py`** | Lightweight orchestrator that detects functional groups and dispatches to rule packs. |
| **`fragmentation_rules_info/`** | Directory of modular rule packs: `universal.py`, `alcohol.py`, `alkene.py`, `amine.py`, `aromatic.py`, `carbonyl.py`, `ester.py`, `ether.py`, `halogen.py`, `phosphorus.py`, `sulfur.py`, plus `base.py` (functional-group detection) and `registry.py` (dispatch). |
| **`hybrid_enumerator.py`** | `HybridEnumerator` — merges rule-based fragments with BDE-driven structural fragments into a single candidate set. Used when PubChem structural data and BDE data are available. |
| **`MLFF_fragmentation/bde_enumerator.py`** | `BDEDrivenEnumerator` — generates fragment formulas by recursively breaking bonds in an RDKit `Mol` object, guided by bond-dissociation energies. |
| **`MLFF_fragmentation/bde_fragmenter.py`** | Low-level RDKit-based bond-breaking engine. Builds a fragmentation tree weighted by BDE softness. |

### Peak Assignment & Scoring

| File | Purpose |
|------|---------|
| **`peak_driven_assignment_engine.py`** | For each observed NIST peak, queries the enumerator for matching fragments, scores them with `fragment_plausibility_score`, applies discriminative confidence normalisation, and emits one `PeakAssignment` per viable candidate. |
| **`assignment_engine.py`** | Legacy assignment engine with multi-match and single-best modes. Imported internally by `peak_driven_assignment_engine.py`. |
| **`peakassignment.py`** | `PeakAssignment` dataclass — container for one peak→fragment assignment (nominal m/z, intensity, exact mass, formula, confidence, rule source). |
| **`fragment.py`** | `Fragment` dataclass — container for a fragment formula with exact mass, nominal mass, DBE, and rule source. |
| **`fps_scoring.py`** | Fragment Plausibility Score (FPS). Heuristic scorer combining cation stability, heteroatom contribution, aromaticity, mass-position, spectral-context, and rule-family priors with tunable weights. |
| **`utils.py`** | Conversion helpers: `convert_assignments()` turns `PeakAssignment` objects into plain dicts, `write_assignments_csv()` exports them. |

### ML Correction

| File | Purpose |
|------|---------|
| **`ml_correction.py`** | `MLCorrectionModel` class. Loads a trained LightGBM booster and feature-name list, builds a rich feature vector (parent descriptors, fragment descriptors, NIST spectral statistics), and predicts a correctness probability for each assignment. |
| **`ml_correction_integration.py`** | Thin integration layer. Instantiates `MLCorrectionModel` once at module level and exposes `apply_ml_correction(assignments, ...)` which mutates each assignment in-place, adding `.ml_prob`. |
| **`ml_correction_model_SIP.txt`** | Serialised LightGBM model file (loaded at runtime). |
| **`ml_feature_importance_SIP.json`** | Feature-name list and importances (used to reconstruct the feature vector order). |

### Post-Hoc Analysis

| File | Purpose |
|------|---------|
| **`analyze_entry_metrics.py`** | Reads `entry_metrics.csv` produced by a validation run. Enriches each entry with chemical descriptors (atom counts, DBE, exact mass, element ratios, BDE statistics, NIST/AML spectrum statistics). Plots precision vs. every descriptor and generates a correlation heatmap. |
| **`plot_fragment_scores.py`** | Reads `fragment_metrics.jsonl`. Produces box-plots, violin plots, and swarm plots of `score`, `ml_prob`, and `hybrid_score` split by `is_correct`. |

### Archived (`old/`)

Deprecated scripts for training, tuning, old runners, and data collection. These are not required to run the validation pipeline but are kept for reference. See `old/` for:
- ML model training (`train_ml_model.py`, `collect_training_data*.py`)
- Scoring parameter tuning (`tune_scoring*.py`, `Tune_score_bo_merged.py`)
- Old batch runners (`runner.py`, `runner_peak_driven.py`)
- Sweep / evaluation scripts (`hybrid_sweep.py`, `hybrid_s.py`, `evaluate_tuned_params.py`)
- Legacy files (`enumerator.py`, `scoring.py`, `JDXParser.py`, `fragmentation_rules_old.py`)
- Training data CSVs and old model/param files

---

## How to Run

### Prerequisites

```
conda activate sumpred          # or your environment name
```

Required packages: `numpy`, `matplotlib`, `pandas`, `seaborn`, `lightgbm`, `rdkit`.

The merged dataset JSON must exist at the path specified by `MERGED_PATH` in `Large_data.py` (default: `/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json`).

### 1. Run the Validation

```bash
cd /home/rat/PycharmProjects/SUMFOR_pred/src/Nist_HR
python validate_hybrid_fragment_recovery.py
```

Or run it directly from PyCharm (right-click → Run).

#### Configuration

Edit the constants at the top of `validate_hybrid_fragment_recovery.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TOL` | `0.0001` Da | Mass tolerance for matching predicted fragments to AML reference peaks. |
| `HYBRID_THR` | `0.75` | Minimum hybrid score for a fragment to be considered "predicted". |
| `RESULTS_OUTPUT["enabled"]` | `True` | Toggle saving per-entry/per-fragment metrics to disk. |
| `RESULTS_OUTPUT["detail"]` | `"entry_and_fragment"` | `"entry"` saves only entry-level metrics; `"entry_and_fragment"` also saves per-fragment records. |
| `RESULTS_OUTPUT["formats"]` | `csv` / `jsonl` / `json` | Output formats for entry metrics, fragment metrics, and run summary respectively. |
| `RESULTS_OUTPUT["out_dir"]` | `"validation_outputs"` | Base directory for output. |
| `RESULTS_OUTPUT["include_fake_spectra"]` | `False` | If `True`, exports predicted fragments as synthetic HR spectra (JSON). |

#### Outputs

After a run, a timestamped directory is created under `validation_outputs/`:

```
validation_outputs/2026-03-26_16-25-23/
├── entry_metrics.csv          # one row per compound (precision, scores, formula)
├── fragment_metrics.jsonl     # one row per predicted fragment (score, ml_prob, is_correct)
├── run_summary.json           # aggregate statistics for the run
├── config_snapshot.json       # exact config used
├── analysis_plots/            # created by analyze_entry_metrics.py
└── fragment_score_plots/      # created by plot_fragment_scores.py
```

Console output includes:
- Progress messages every 100 entries
- Precision statistics (mean, median, std, quartiles)
- Fragment count statistics
- Three matplotlib plots (precision distribution, fragment count distribution, precision vs. fragment count scatter)

### 2. Analyse Entry-Level Metrics

```bash
python analyze_entry_metrics.py validation_outputs/<run_id>/entry_metrics.csv
```

Or run from PyCharm (it will prompt for the path interactively).

This produces ~50 scatter/box plots in `analysis_plots/` showing precision vs. chemical descriptors (atom counts, DBE, exact mass, halogen counts, BDE statistics, NIST/AML peak counts, etc.) plus a correlation heatmap.

### 3. Analyse Fragment-Level Scores

```bash
python plot_fragment_scores.py validation_outputs/<run_id>/fragment_metrics.jsonl
```

Or run from PyCharm (uses the hardcoded default path if no argument is given).

This produces box plots, violin plots, and swarm plots in `fragment_score_plots/` showing how `score`, `ml_prob`, and `hybrid_score` distributions differ between correct and incorrect fragment predictions.

---

## Key Concepts

### Hybrid Score

```
hybrid_score = α × physics_score + (1 − α) × ml_prob
```

where `α = 0.4` (defined in `Large_data.py`). An override rule applies when the physics score is very low (`< 0.1`) but the ML model is confident (`ml_prob ≥ 0.8`): the hybrid score is set to `ml_prob` directly.

### Precision

```
precision = correct_fragments / predicted_fragments
```

A fragment is "predicted" if its `hybrid_score ≥ HYBRID_THR`. It is "correct" if its exact mass matches any AML reference peak within `TOL` Da.

### Top-K Cap

After hybrid scoring, only the top **20** fragments (by hybrid score) are kept per entry (configured in `Large_data.py`).

### Enumerator Selection

For each entry, `Large_data.py` checks whether PubChem structural data (SMILES/InChI) and BDE bond data are available:
- **Yes** → `HybridEnumerator` (rules + BDE-driven structural fragmentation)
- **No** → `PeakDrivenEnumerator` (rule-based only, fallback)

---

## Pipeline Stages (per entry)

1. **Load data** — NIST EI spectrum + AML HR spectrum + parent formula + PubChem/BDE metadata from the merged JSON.
2. **Filter NIST peaks** — Keep only peaks with relative intensity ≥ 5 %.
3. **Enumerate fragments** — Generate candidate sub-formulas for each NIST peak using rules and/or BDE-driven fragmentation (up to depth 5).
4. **Assign peaks** — Score each candidate with the Fragment Plausibility Score (FPS), apply discriminative normalisation, emit one `PeakAssignment` per viable candidate.
5. **ML correction** — A pre-trained LightGBM model predicts a correctness probability (`ml_prob`) for each assignment based on a rich feature vector.
6. **Hybrid score** — Combine physics score and ML probability.
7. **Top-K filter** — Keep only the 20 highest-scoring fragments.
8. **Validate** — Compare predicted fragment masses to AML reference peaks; compute precision.
9. **Save** — Write entry-level and fragment-level metrics to disk (if enabled).
