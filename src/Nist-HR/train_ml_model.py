# train_ml_model.py

import pandas as pd
import lightgbm as lgb
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)

TRAIN_CSV = "training_fragments.csv"
MODEL_OUT = "ml_correction_model.txt"
FEATURES_OUT = "ml_feature_importance.json"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)

# Remove rows with missing critical values
df = df.dropna(subset=["frag_mass", "confidence", "parent_mass"])


# ------------------------------------------------------------
# Train/validation split by entry_id
# ------------------------------------------------------------
unique_ids = df["entry_id"].unique()
train_ids, val_ids = train_test_split(
    unique_ids, test_size=0.2, random_state=42
)

print(f"Total unique entries: {len(unique_ids)}")


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

# Multi-class string → multi-hot encoding
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

# Expand class labels and append to df
class_features = expand_classes(df["classes"])
df = pd.concat([df, class_features], axis=1)

# Categorical: rule_family
df["rule_family"] = df["rule_family"].fillna("none").astype("category")

# Re-split after adding features / dtypes
train_df = df[df["entry_id"].isin(train_ids)]
val_df   = df[df["entry_id"].isin(val_ids)]

print(f"Training fragments: {len(train_df)}")
print(f"Validation fragments: {len(val_df)}")


# ------------------------------------------------------------
# Select features
# ------------------------------------------------------------
feature_cols = [

    # parent
    "parent_mass", "parent_dbe", "n_C", "n_H", "n_O", "n_N", "n_halogen",

    # NIST global
    "nist_n_peaks", "nist_base_mz", "nist_base_intensity",
    "nist_entropy", "nist_peak_density",
    "nist_intensity_mean", "nist_intensity_std",
    "nist_highmass_fraction", "nist_lowmass_fraction",
    "has_peak_77", "has_peak_91", "has_peak_105",
    "has_cl_pattern", "has_br_pattern",

    # peak-level
    "peak_nominal_mz", "peak_intensity", "peak_rel_intensity",
    "local_intensity_mz", "local_intensity_mz_minus1",
    "local_intensity_mz_plus1", "local_intensity_mz_minus14",
    "local_intensity_mz_plus14", "local_peak_density",

    # fragment-level
    "frag_mass", "frag_dbe", "mass_fraction", "confidence",

    # categorical
    "rule_family",
]

# Add class_* columns
feature_cols += [c for c in df.columns if c.startswith("class_")]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_val = val_df[feature_cols]
y_val = val_df["label"]

# Categorical features for LightGBM
categorical_features = ["rule_family"]


# ------------------------------------------------------------
# LightGBM model
# ------------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-5,
    num_leaves=164,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=categorical_features,
)


# ------------------------------------------------------------
# Validation metrics
# ------------------------------------------------------------
val_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, val_pred)
ap  = average_precision_score(y_val, val_pred)
acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
f1  = f1_score(y_val, (val_pred > 0.5).astype(int))

print("\nValidation performance:")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 score: {f1:.4f}")


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------
model.booster_.save_model(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")


# ------------------------------------------------------------
# Save feature importance
# ------------------------------------------------------------
importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
with open(FEATURES_OUT, "w") as f:
    json.dump(importance, f, indent=2)

print(f"Saved feature importance to {FEATURES_OUT}")