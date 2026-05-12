# train_ml_model.py

import pandas as pd
import lightgbm as lgb
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)

TRAIN_CSV = "training_fragments_SIP_hybrid_BDE.csv"
MODEL_OUT = "ml_correction_model_SIP.txt"
FEATURES_OUT = "ml_feature_importance_SIP.json"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)

# Remove rows with missing critical values
df = df.dropna(subset=["frag_mass", "confidence", "parent_mass"])

print(f"Loaded {len(df):,} rows from {TRAIN_CSV}")


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


# Expand class labels and append to df
class_features = expand_classes(df["classes"])
df = pd.concat([df, class_features], axis=1)

# Categorical: rule_family
df["rule_family"] = df["rule_family"].fillna("none").astype("category")


# ------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------
base_features = [
    "parent_mass", "parent_dbe",
    "n_C", "n_H", "n_O", "n_N",
    "n_S", "n_P",
    "n_F", "n_Cl", "n_Br", "n_I",
    "n_halogen",
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
    "peak_nominal_mz",
    "peak_intensity",
    "peak_rel_intensity",
    "local_intensity_mz",
    "local_intensity_mz_minus1",
    "local_intensity_mz_plus1",
    "local_intensity_mz_minus14",
    "local_intensity_mz_plus14",
    "local_peak_density",
    "frag_mass",
    "frag_dbe",
    "mass_fraction",
    "confidence",
    "frag_n_C", "frag_n_H", "frag_n_O", "frag_n_N",
    "frag_n_S", "frag_n_P",
    "frag_n_F", "frag_n_Cl", "frag_n_Br", "frag_n_I",
    "frag_n_halogen",
    "rule_family",
]

new_feature_candidates = [
    "is_bde_fragment",
    "n_bde_steps",
    "n_rule_steps",
    "path_length",
    "contains_aromatic_rule",
    "contains_neutral_loss",
    "contains_rearrangement",
    "mass_defect",
    "H_to_C_ratio",
    "N_to_C_ratio",
    "O_to_C_ratio",
    "is_common_ei_ion",
    "distance_to_nearest_peak",
    "intensity_of_nearest_peak",
    "within_1Da",
    "within_2Da",
]

new_features = [f for f in new_feature_candidates if f in df.columns]

feature_cols = base_features + new_features
feature_cols += [c for c in df.columns if c.startswith("class_")]

print(f"Total features used: {len(feature_cols)}")


# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------
X = df[feature_cols]
y = df["label"]
groups = df["entry_id"]

categorical_features = ["rule_family"]


# ------------------------------------------------------------
# K-Fold Cross Validation (Stratified by label, grouped by molecule)
# ------------------------------------------------------------
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
feature_importances = np.zeros(len(feature_cols))

fold_idx = 1

for train_idx, val_idx in kf.split(X, y, groups):
    print(f"\n=== Fold {fold_idx} ===")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.03,
        max_depth=-5,
        num_leaves=196,
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

    val_pred = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_pred)
    ap  = average_precision_score(y_val, val_pred)
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    f1  = f1_score(y_val, (val_pred > 0.5).astype(int))

    print(f"AUC: {auc:.4f}")
    print(f"AP:  {ap:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"F1:  {f1:.4f}")

    fold_metrics.append({"auc": auc, "ap": ap, "acc": acc, "f1": f1})
    feature_importances += model.feature_importances_

    fold_idx += 1


# ------------------------------------------------------------
# Average metrics across folds
# ------------------------------------------------------------
avg_metrics = {
    "auc": np.mean([m["auc"] for m in fold_metrics]),
    "ap":  np.mean([m["ap"]  for m in fold_metrics]),
    "acc": np.mean([m["acc"] for m in fold_metrics]),
    "f1":  np.mean([m["f1"]  for m in fold_metrics]),
}

print("\n=== Average CV Performance ===")
for k, v in avg_metrics.items():
    print(f"{k.upper()}: {v:.4f}")


# ------------------------------------------------------------
# Train final model on all data
# ------------------------------------------------------------
final_model = lgb.LGBMClassifier(
    n_estimators=1600,
    learning_rate=0.03,
    max_depth=-5,
    num_leaves=196,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

final_model.fit(
    X, y,
    categorical_feature=categorical_features,
)

final_model.booster_.save_model(MODEL_OUT)
print(f"\nSaved final model to {MODEL_OUT}")


# ------------------------------------------------------------
# Save averaged feature importance
# ------------------------------------------------------------
importance = dict(zip(feature_cols, feature_importances.tolist()))
with open(FEATURES_OUT, "w") as f:
    json.dump(importance, f, indent=2)

print(f"Saved feature importance to {FEATURES_OUT}")