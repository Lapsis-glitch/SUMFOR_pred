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
├── ml_correction_model_SIP.txt            ← trained model weights (no held-out entries)
├── ml_calibrator_v2.pkl                   ← isotonic calibrator for ml_prob
├── ml_feature_importance_SIP.json         ← feature names
├── train_ml_model_v2.py                   ← retrain ml_correction_model_SIP.txt
├── collect_training_data_v2.py            ← regenerate training_fragments_*_v2.csv
│
├── final_decision_calibrator.py           ← post-hoc final-decision probability layer
├── final_decision_calibrator.pkl          ← trained final-decision artifact
├── train_final_decision_calibrator.py     ← retrain the final-decision calibrator
│
├── conformal_calibrator.py                ← cross-conformal p-value layer (TP null)
├── conformal_calibration.pkl              ← frozen calibration table (K-fold OOF)
├── build_conformal_calibration.py         ← build/refresh the conformal calibration table
│
├── heldout_split.py                       ← shared held-out test loader
├── heldout_test_entries.json              ← 310 entry_ids reserved as the clean test set
├── generate_heldout_test_set.py           ← one-shot script that wrote heldout_test_entries.json
│
├── validate_hybrid_fragment_recovery_precision.py  ← precision-oriented validator (strict gates + calibrator + rescue)
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
validate_hybrid_fragment_recovery_precision.py
              │                   strict gates (hybrid / ML / posterior / acceptance)
              ▼
   FinalDecisionCalibrator        (decision_prob — "is this top-1 correct?")
              │
              ▼
   ConformalCalibrator            (conformal_p — "how does it rank vs. known TPs?")
              │
     ┌────────┴────────────┐
     │                     │
     ▼                     ▼
 entry_metrics      fragment_metrics + fake_spectra_strict/*.json
     │                     │
     ▼                     ▼
analyze_entry_metrics.py   plot_fragment_scores.py   (post-hoc analysis)
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
| **`ml_correction_model_SIP.txt`** | Serialised LightGBM model file (loaded at runtime). Retrained on 2026-05-12 with the held-out test slice excluded (see "Clean-Room Test Set" below). The previous v1 production model is preserved as `ml_correction_model_SIP.txt.bak.2026-05-12_clean_test_setup`. |
| **`ml_calibrator_v2.pkl`** | Isotonic post-calibration of the LightGBM raw probabilities. Used by `ml_correction.py` to convert raw scores into `ml_prob`. |
| **`ml_feature_importance_SIP.json`** | Feature-name list and importances (used to reconstruct the feature vector order). |
| **`train_ml_model_v2.py`** | Trainer for the LightGBM model. Reads `training_fragments_SIP_hybrid_BDE_v2.csv`, drops entries listed in `heldout_test_entries.json`, splits by `entry_id` (70/15/15), fits the model, runs isotonic calibration, and overwrites `ml_correction_model_SIP.txt`. |
| **`collect_training_data_v2.py`** | Regenerates the training CSV by running the full enumeration + assignment pipeline on every non-heldout entry. Multi-process. |
| **`final_decision_calibrator.py`** | Loader + feature-builder + scorer for the post-hoc final-decision calibrator. The calibrator gates the precision validator's top-1-per-peak export and powers the calibrated rescue / top-up stages. |
| **`final_decision_calibrator.pkl`** | Trained LightGBM (+ optional isotonic) artifact. |
| **`train_final_decision_calibrator.py`** | Trains `final_decision_calibrator.pkl` from a precision-validator `fragment_metrics.jsonl`. Drops held-out entries, then runs stratified group K-fold CV plus a 15% internal holdout. |
| **`conformal_calibrator.py`** | `ConformalCalibrator` — frozen K-fold OOF table of (is_correct, decision_prob, parent-class stratum) used to emit per-fragment conformal p-values testing the TP null (global pooled + Mondrian class-conditional). Also provides `combine_p_values()` for spectrum-level aggregation (HMP / Fisher / Bonferroni-min / raw min). |
| **`conformal_calibration.pkl`** | Frozen calibration table (one row per train-pool top1-per-peak fragment). Built by `build_conformal_calibration.py`. Loaded once by the precision validator. |
| **`build_conformal_calibration.py`** | Runs stratified-group K-fold CV over the latest train-pool `fragment_metrics.jsonl`, captures each row's out-of-fold `decision_prob`, and writes `conformal_calibration.pkl`. Supports `--verify-on <test_only fragment_metrics.jsonl>` to print empirical coverage at α∈{0.01, 0.05, 0.10, 0.20}. Re-run after any change to the final-decision calibrator's model spec or training data. |
| **`heldout_split.py`** | Tiny shared loader: `load_heldout_test_ids()`, `split_train_pool()`. Used by both trainers, the collector, and the precision validator's `SUMFOR_RUN_MODE` switch. |
| **`heldout_test_entries.json`** | The reserved 310 entry_ids (15% of the corpus, `seed=7`). No model trains on these. |
| **`generate_heldout_test_set.py`** | One-shot script that produced `heldout_test_entries.json`. Run with `--force` to regenerate — but doing so **invalidates every model that was trained against the previous split**. |

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

### 4. Run the Precision Validator (and Clean-Room Test)

`validate_hybrid_fragment_recovery_precision.py` is the precision-oriented sibling of the baseline validator. It applies strict per-peak gates (hybrid threshold, ML floor, peak posterior, top1-vs-top2 margin, acceptance score, final-decision calibrator), an FDR-budgeted rescue stage, and a calibrated top-up stage. Outputs land in `validation_outputs_precision/<run_id>/`. Configuration lives in the `STRICT_SELECTION`, `FINAL_DECISION_LAYER`, and `CONFORMAL_LAYER` dicts at the top of the file.

```bash
cd src/Nist_HR
python validate_hybrid_fragment_recovery_precision.py                      # all entries
SUMFOR_RUN_MODE=train_pool python validate_hybrid_fragment_recovery_precision.py   # only entries the models trained on
SUMFOR_RUN_MODE=test_only  python validate_hybrid_fragment_recovery_precision.py   # only the 310 held-out test entries
```

The `run_id` is suffixed with the mode (`..._train_pool`, `..._test_only`) so output dirs are unambiguous, and `run_summary.json` records both `run_mode` and `entries_in_run_mode`.

If `conformal_calibration.pkl` is missing or `CONFORMAL_LAYER["enabled"] = False`, the run still works — it just won't emit `conformal_p_*` columns or the `spectrum_confidence` block.

### 5. Build the Conformal Calibration Table

The conformal layer needs a frozen calibration table of out-of-fold (OOF) calibrator predictions. Build it once after every retraining of `final_decision_calibrator.pkl`:

```bash
# Default: latest train_pool fragment_metrics.jsonl + lightgbm + 5-fold CV
python build_conformal_calibration.py

# With a coverage sanity check against a test_only run
python build_conformal_calibration.py \
    --verify-on validation_outputs_precision/<test_only-run>/fragment_metrics.jsonl
```

What it does: re-runs stratified-group K-fold CV (`group=entry_id`) of the calibrator over the train-pool top1-per-peak rows so every row's predicted probability comes from a fold that did not see its entry. The artifact stores `(is_correct, oof_prob, parent_class_stratum)` triples. Strata with fewer than 50 positive calibration rows auto-fall-back to the global pool at inference.

Cost: a few seconds per fold on a sane `--n-jobs` (defaults to 4 — `-1` thrashes the OpenMP threads and is much slower in practice). Total wall-clock ~30–60 s for ~39k rows × 5 folds on this hardware.

---

## Clean-Room Test Set

A 15% slice of `MERGED` (310 of 2,066 entry_ids) is reserved in `heldout_test_entries.json` and never seen by training. The split was generated once with `seed=7` — distinct from the `seed=42` used inside both trainers — so it is independent of any internal trainer split.

Who honors it:

| File | Behaviour |
|------|-----------|
| `collect_training_data_v2.py` | Skips held-out entries when generating the training CSV. |
| `train_ml_model_v2.py` | Drops held-out rows after loading the CSV, before its internal 70/15/15 split. (Defense-in-depth — the CSV already excludes them.) |
| `train_final_decision_calibrator.py` | Drops held-out rows from `fragment_metrics.jsonl` before its pool/holdout split. |
| `build_conformal_calibration.py` | Drops held-out rows from `fragment_metrics.jsonl` before K-fold CV, so the calibration table is built from train-pool fragments only. |
| `validate_hybrid_fragment_recovery_precision.py` | Filters `ENTRY_IDS` based on `SUMFOR_RUN_MODE` (see above). |

Latest clean-room evaluation (`validation_outputs_precision/2026-05-12_10-22-29_test_only/`):

| Metric | Value |
|--------|-------|
| Entries with strict predictions | 271 / 310 (87.4%) |
| Mean precision | 0.9487 |
| Median precision | 1.0000 |
| Mean fragments selected / spectrum | 4.94 |
| Mean fragments correct / spectrum | 4.70 |

Compare to the most recent full-corpus tuning (`98.28% / median 7`): the ~3-point precision drop and the smaller spectrum size on never-seen entries is the honest in-distribution generalisation gap.

**Regenerating the split.** Don't. Running `generate_heldout_test_set.py --force` will pick a different 15% slice and silently invalidate every artifact (`ml_correction_model_SIP.txt`, `final_decision_calibrator.pkl`, all backups dated `2026-05-12_clean_test_setup`). If a fresh split is genuinely needed, retrain both models afterwards and produce a new clean-room run.

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

### Conformal Prediction Layer

The precision validator sits on a stack of five layers, each answering a sharper question than the one before:

| Layer | Question it answers | Output |
|-------|---------------------|--------|
| **Enumerator** | What formulas could plausibly land on each peak? | candidate list per peak |
| **Scorer** (FPS + ML correction) | How well does each candidate match physics + ML? | `physics_score`, `ml_prob`, `hybrid_score` |
| **Strict selection** | Which one wins per peak — and is it confident enough? | one-best-per-peak after gates |
| **Final-decision calibrator** | Given everything above, what's the probability this is a TP? | `decision_prob` ∈ [0, 1] |
| **Conformal layer** | How does that probability *compare to known TPs*? | `conformal_p` ∈ [0, 1] |

The first four are *model outputs* — they tell you what the model believes. The conformal layer is a *statistical statement* about how trustworthy that belief is, calibrated against held-out ground truth.

**Intuition.** Think of `conformal_calibration.pkl` as a frozen reference deck of TP confidence scores. We built it once by running 5-fold CV on the train-pool fragments — every fragment got a calibrator probability from a fold that had never seen it. We kept only the rows that were *actually* true positives against AML. Their probabilities form the deck.

At inference, a new fragment gets `decision_prob = 0.87`. We ask: *in our deck of known TPs, what fraction had a probability ≤ 0.87?* If 70% of known TPs scored worse than this, then the fragment's **conformal p-value is ~0.70** — it looks like a typical TP. A high p ⇒ "indistinguishable from known TPs." A low p ⇒ "looks worse than nearly every known TP, so probably an FP."

**Mondrian (class-aware).** Separate decks per parent-class group (`aromatic`, `halogenated`, `nitrogenous`, `oxygenated`, `phosphorus`, `sulfur`, `other`) so a halogenated fragment is compared only to halogenated TPs. Small classes (< 50 positives) auto-fall-back to the global pool — the `conformal_stratum_used` field records which deck was actually used.

**Per-spectrum confidence.** Per-fragment p-values are combined into a single spectrum-level number. The validator emits four flavours and you pick the one that suits your downstream use:

| Method | Best for | Caveat |
|--------|----------|--------|
| `conformal_spectrum_p_hmp` (harmonic-mean p) | Headline confidence — robust under positive dependence between fragments of the same molecule. | Asymptotic null; tightest of the three valid options. |
| `conformal_spectrum_p_bonferroni` (`min(1, k·p_min)`) | A conservative bound valid under arbitrary dependence. | Often loose, but never wrong. |
| `conformal_spectrum_p_fisher` (Fisher combined) | Reference / diagnostics. | Assumes independence; anti-conservative on correlated fragments. |
| `conformal_spectrum_p_min` | Raw minimum p-value across fragments. | No multiplicity correction; useful for sorting only. |

Low spectrum p ⇒ at least one fragment looks suspicious; high spectrum p ⇒ the whole prediction looks like a typical good prediction.

**Optional gate.** Setting `CONFORMAL_LAYER["p_value_gate"] = 0.05` drops strictly-selected fragments whose conformal p falls below 0.05, with rejection reason `below_conformal_p`. Off by default — the layer is purely informational unless you opt in.

**Empirical coverage** on the held-out 310 entries (`fragment_metrics.jsonl` rows where `rank_within_peak == 1` and `is_correct == True`):

| α    | observed reject rate | nominal |
|------|---------------------|---------|
| 0.01 | 0.78%               | ≤ 1%    |
| 0.05 | 3.86%               | ≤ 5%    |
| 0.10 | 8.92%               | ≤ 10%   |
| 0.20 | 19.02%              | ≤ 20%   |

Discrimination on the same retained pool: true TPs have mean p ≈ 0.50 (theoretical uniform), true FPs have mean p ≈ 0.11 (52% at p ≤ 0.05).

---

## Pipeline Stages (per entry)

1. **Load data** — NIST EI spectrum + AML HR spectrum + parent formula + PubChem/BDE metadata from the merged JSON.
2. **Filter NIST peaks** — Keep only peaks with relative intensity ≥ 5 %.
3. **Enumerate fragments** — Generate candidate sub-formulas for each NIST peak using rules and/or BDE-driven fragmentation (up to depth 5).
4. **Assign peaks** — Score each candidate with the Fragment Plausibility Score (FPS), apply discriminative normalisation, emit one `PeakAssignment` per viable candidate.
5. **ML correction** — A pre-trained LightGBM model predicts a correctness probability (`ml_prob`) for each assignment based on a rich feature vector.
6. **Hybrid score** — Combine physics score and ML probability.
7. **Top-K filter** — Keep only the 20 highest-scoring fragments.
8. **(precision validator only) Strict selection** — One-best-per-peak after hybrid / ML / peak-posterior / acceptance gates, plus an FDR-budgeted rescue stage and a calibrated top-up stage.
9. **(precision validator only) Final-decision probability** — `FinalDecisionCalibrator` attaches `decision_prob` to each top-1 candidate.
10. **(precision validator only) Conformal p-value** — `ConformalCalibrator` attaches `conformal_p_global` and `conformal_p_mondrian` per fragment; combines them into per-spectrum `conformal_spectrum_p_{hmp,fisher,bonferroni,min}`.
11. **Validate** — Compare predicted fragment masses to AML reference peaks; compute precision.
12. **Save** — Write entry-level and fragment-level metrics to disk (if enabled). Precision runs additionally write `fake_spectra_strict/entry_<id>.json` with per-fragment p-values and a top-level `spectrum_confidence` block.
