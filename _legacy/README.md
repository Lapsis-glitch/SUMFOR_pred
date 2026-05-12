# _legacy/

Archive of code and model artifacts moved out of the working tree on
**2026-05-12** during the cleanup pass on the `Cleanup` branch.

Nothing in here is on a live code path. Do not import from `_legacy/`,
do not run scripts from inside it — internal paths assume the original
locations (`src/`, `src/Nist_HR/`, project root) and are likely broken.
Files are preserved for historical reference only.

## Layout

### `torch_encoder/`
Ex-`src/` PyTorch encoder/decoder project: a SMILES-from-spectrum
sequence model. Unrelated to the current Nist_HR fragment-prediction
pipeline. Contents:
- Python sources: `BeamDecoder.py`, `DatasetBuilder.py`, `FormulaUtils.py`,
  `JDX.py`, `NISTMS.py`, `SpecRep.py`, `Trainer.py`, `data_extract.py`,
  `dataset.py`, `model.py`, `rulecheck.py`, `train.py`, `utils.py`
- Model checkpoints: `0.pt`, `best_mass_encoder.pt`,
  `checkpoint_mass_encoder.pt`, `checkpoint_mass_encoder1.pt`

### `root_scratchpad/`
- `main.py` — project-root scratchpad noted in CLAUDE.md as
  "unrelated/legacy scratchpad". Not part of any pipeline.

### `nist_hr/old/` and `nist_hr/bk/`
Pre-existing legacy folders that were already inside `src/Nist_HR/`.
Relocated unchanged. `old/` contains an earlier generation of the
pipeline (separate `enumerator.py`, `scoring.py`, `runner.py`, an
earlier `train_ml_model.py`, the v1 `training_fragments_SIP.csv`, etc.)
plus an old set of fake-spectra exports. `bk/` holds a manual backup
of the v1 ML correction model files.

### `nist_hr/archived_models/`
Older model artifacts and timestamped `.bak` copies kept around the
2026-05-11 / 2026-05-12 retraining cycle. The currently-loaded
production artifacts (`ml_correction_model_SIP.txt`,
`final_decision_calibrator.pkl`, `conformal_calibration.pkl`,
`ml_calibrator_v2.pkl`, `ml_feature_importance_SIP.json`) stayed in
`src/Nist_HR/`. Files in this folder include:

- `ml_correction_model_SIP_v1.txt` — the long-running v1 ML model
  (Feb 2026). Replaced 2026-05-12 by a clean-room retrain of v1's
  feature set, which is what `ml_correction_model_SIP.txt` now is.
- `ml_correction_model_SIP_v2.txt` — earlier v2 feature-expansion
  retrain that underperformed v1 (recorded in CLAUDE.md as the reason
  the v2 trainer kept the v1-style features).
- `*_v1.json` / `*_v2.json` — paired feature-importance dumps.
- `*.bak.2026-05-12_clean_test_setup` — pre-clean-test-set snapshots
  of every artifact, taken just before retraining without the
  held-out entries.
- `*.bak.20260511`, `*.bak.20260512_lr` — earlier checkpoints from
  the calibrator tuning runs that landed at commit `ebf891f`.

## Restoring a file

If something here is needed again, copy (don't move) it back into the
appropriate live location, then update the loader path in code. Avoid
moving the file back blindly — the `.pkl` schemas may not match the
current `FinalDecisionCalibrator` / `ConformalCalibrator` class shapes.
