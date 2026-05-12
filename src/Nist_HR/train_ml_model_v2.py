# train_ml_model_v2.py
#
# Retrain LightGBM on regenerated training data (collect_training_data_v2.py).
#
# Keeps `confidence` as a feature (the v1 model needs it).
# Adds new BDE/pathway + structural features.
# Adds isotonic calibration for smoother probability outputs.
#
# Outputs overwrite the v1 model files so the pipeline picks them up
# automatically.

import pandas as pd
import lightgbm as lgb
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
import joblib

from heldout_split import load_heldout_test_ids

TRAIN_CSV = "training_fragments_SIP_hybrid_BDE_v2.csv"
MODEL_OUT = "ml_correction_model_SIP.txt"
CALIBRATOR_OUT = "ml_calibrator_v2.pkl"
FEATURES_OUT = "ml_feature_importance_SIP.json"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)
df = df.dropna(subset=["frag_mass", "confidence", "parent_mass"])

print(f"Total rows: {len(df)}")
print(f"Positive rate: {df['label'].mean():.4f}")
print(f"Unique entries: {df['entry_id'].nunique()}")

# ------------------------------------------------------------
# Drop the shared held-out test slice before any split
# ------------------------------------------------------------
heldout_ids = load_heldout_test_ids()
if heldout_ids:
    entry_id_str = df["entry_id"].astype(str)
    mask_heldout = entry_id_str.isin(heldout_ids)
    n_heldout_rows = int(mask_heldout.sum())
    n_heldout_entries = int(entry_id_str[mask_heldout].nunique())
    df = df.loc[~mask_heldout].reset_index(drop=True)
    print(
        f"Excluded held-out test slice: {n_heldout_rows} rows from "
        f"{n_heldout_entries} entries (out of {len(heldout_ids)} reserved IDs)."
    )
    print(f"Rows after exclusion: {len(df)}")
else:
    print("No heldout_test_entries.json found — training on the full dataset (NO clean test slice).")


# ------------------------------------------------------------
# Train/validation/calibration split by entry_id
# ------------------------------------------------------------
unique_ids = df["entry_id"].unique()

# 70% train, 15% calibration, 15% validation
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.30, random_state=42)
cal_ids, val_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

print(f"Train entries: {len(train_ids)}")
print(f"Calibration entries: {len(cal_ids)}")
print(f"Validation entries: {len(val_ids)}")


# ------------------------------------------------------------
# Multi-class string → multi-hot encoding
# ------------------------------------------------------------
def expand_classes(series):
    all_classes = set()
    for s in series:
        if isinstance(s, str):
            for c in s.split(";"):
                all_classes.add(c)
    all_classes = sorted(all_classes)

    out = pd.DataFrame(0, index=series.index, columns=[f"class_{c}" for c in all_classes])
    for idx, s in series.items():
        if isinstance(s, str):
            for c in s.split(";"):
                out.at[idx, f"class_{c}"] = 1
    return out


class_features = expand_classes(df["classes"])
df = pd.concat([df, class_features], axis=1)
df["rule_family"] = df["rule_family"].fillna("none").astype("category")


# ------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------
feature_cols = [
    # parent-level descriptors
    "parent_mass", "parent_dbe",
    "n_C", "n_H", "n_O", "n_N",
    "n_S", "n_P",
    "n_F", "n_Cl", "n_Br", "n_I",
    "n_halogen",

    # NIST global descriptors
    "nist_n_peaks",
    "nist_base_mz",
    "nist_base_intensity",
    "nist_entropy",
    "nist_peak_density",
    "nist_intensity_mean",
    "nist_intensity_std",
    "nist_highmass_fraction",
    "nist_lowmass_fraction",
    "has_peak_77",
    "has_peak_91",
    "has_peak_105",
    "has_cl_pattern",
    "has_br_pattern",
    "has_i_pattern",

    # peak-level descriptors
    "peak_nominal_mz",
    "peak_intensity",
    "peak_rel_intensity",
    "local_intensity_mz",
    "local_intensity_mz_minus1",
    "local_intensity_mz_plus1",
    "local_intensity_mz_minus14",
    "local_intensity_mz_plus14",
    "local_peak_density",

    # fragment-level descriptors
    "frag_mass",
    "frag_dbe",
    "mass_fraction",
    "confidence",          # physics score — kept

    # fragment element counts
    "frag_n_C", "frag_n_H", "frag_n_O", "frag_n_N",
    "frag_n_S", "frag_n_P",
    "frag_n_F", "frag_n_Cl", "frag_n_Br", "frag_n_I",
    "frag_n_halogen",

    # BDE / pathway features (new)
    "is_bde_fragment",
    "n_bde_steps", "n_rule_steps", "path_length",
    "contains_aromatic_rule", "contains_neutral_loss", "contains_rearrangement",

    # Fragment structural features (new)
    "mass_defect",
    "H_to_C_ratio", "N_to_C_ratio", "O_to_C_ratio",
    "is_common_ei_ion",

    # Peak proximity (new)
    "distance_to_nearest_peak", "intensity_of_nearest_peak",
    "within_1Da", "within_2Da",

    # categorical
    "rule_family",
]

# Add class_* columns
feature_cols += [c for c in df.columns if c.startswith("class_")]

# Verify all columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"WARNING: missing columns (will be skipped): {missing}")
    feature_cols = [c for c in feature_cols if c in df.columns]


# Split
train_df = df[df["entry_id"].isin(train_ids)]
cal_df   = df[df["entry_id"].isin(cal_ids)]
val_df   = df[df["entry_id"].isin(val_ids)]

X_train, y_train = train_df[feature_cols], train_df["label"]
X_cal, y_cal     = cal_df[feature_cols], cal_df["label"]
X_val, y_val     = val_df[feature_cols], val_df["label"]

print(f"\nTraining fragments:    {len(X_train)} (pos={y_train.sum()})")
print(f"Calibration fragments: {len(X_cal)} (pos={y_cal.sum()})")
print(f"Validation fragments:  {len(X_val)} (pos={y_val.sum()})")

categorical_features = ["rule_family"]


# ------------------------------------------------------------
# LightGBM model — v1-like power + regularisation
# ------------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=-1,             # unlimited (same as v1)
    num_leaves=196,           # same as v1
    min_child_samples=30,     # slightly more than v1 default
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_cal, y_cal)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100)],
    categorical_feature=categorical_features,
)


# ------------------------------------------------------------
# Isotonic calibration on the calibration set
# ------------------------------------------------------------
cal_raw = model.predict_proba(X_cal)[:, 1]
iso_reg = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
iso_reg.fit(cal_raw, y_cal)

print(f"\nCalibration set raw probs: mean={cal_raw.mean():.4f}  median={np.median(cal_raw):.4f}")
cal_calibrated = iso_reg.predict(cal_raw)
print(f"Calibration set calibrated: mean={cal_calibrated.mean():.4f}  median={np.median(cal_calibrated):.4f}")


# ------------------------------------------------------------
# Validation metrics (calibrated)
# ------------------------------------------------------------
val_raw = model.predict_proba(X_val)[:, 1]
val_pred = iso_reg.predict(val_raw)

auc = roc_auc_score(y_val, val_pred)
ap  = average_precision_score(y_val, val_pred)
acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
f1  = f1_score(y_val, (val_pred > 0.5).astype(int))
brier = brier_score_loss(y_val, val_pred)
ll = log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7))

print("\nValidation performance (calibrated):")
print(f"  AUC:                {auc:.4f}")
print(f"  Average Precision:  {ap:.4f}")
print(f"  Accuracy:           {acc:.4f}")
print(f"  F1 score:           {f1:.4f}")
print(f"  Brier score:        {brier:.4f}")
print(f"  Log loss:           {ll:.4f}")

# Score distribution
print(f"\nCalibrated score distribution (val set):")
for lo in np.arange(0.0, 1.01, 0.10):
    hi = lo + 0.10
    n = np.sum((val_pred >= lo) & (val_pred < hi))
    print(f"  [{lo:.1f}, {hi:.1f}): {n:5d} ({100*n/len(val_pred):.1f}%)")

print(f"\nRaw (uncalibrated) score distribution (val set):")
for lo in np.arange(0.0, 1.01, 0.10):
    hi = lo + 0.10
    n = np.sum((val_raw >= lo) & (val_raw < hi))
    print(f"  [{lo:.1f}, {hi:.1f}): {n:5d} ({100*n/len(val_raw):.1f}%)")


# ------------------------------------------------------------
# Save model + calibrator
# ------------------------------------------------------------
model.booster_.save_model(MODEL_OUT)
print(f"\nSaved model to {MODEL_OUT}")

joblib.dump(iso_reg, CALIBRATOR_OUT)
print(f"Saved calibrator to {CALIBRATOR_OUT}")


# ------------------------------------------------------------
# Save feature importance
# ------------------------------------------------------------
importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
with open(FEATURES_OUT, "w") as f:
    json.dump(importance, f, indent=2)

print(f"Saved feature importance to {FEATURES_OUT}")
print(f"\nTop 15 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:15]):
    print(f"  {i+1:2d}. {feat:35s} {imp:6d}")

