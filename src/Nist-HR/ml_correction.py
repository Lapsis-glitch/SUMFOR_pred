# ml_correction.py

import math
import json
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from formula import Formula
from chemistry import exact_mass, dbe
from chemical_classification import classify_molecule

MODEL_OUT = "ml_correction_model.txt"
FEATURES_IMPORTANCE = "ml_feature_importance.json"


# ------------------------------------------------------------
# Load model + feature names
# ------------------------------------------------------------
class MLCorrectionModel:
    def __init__(self,
                 model_path: str = MODEL_OUT,
                 importance_path: str = FEATURES_IMPORTANCE):
        self.booster = lgb.Booster(model_file=model_path)

        # feature names from importance file (keys)
        with open(importance_path, "r") as f:
            importance = json.load(f)
        self.feature_names = list(importance.keys())

    # --------------------------------------------------------
    # NIST global descriptors
    # --------------------------------------------------------
    @staticmethod
    def nist_entropy(intensities):
        total = sum(intensities)
        if total == 0:
            return 0.0
        p = [i / total for i in intensities]
        return -sum(pi * math.log(pi) for pi in p if pi > 0)

    @staticmethod
    def peak_density(n_peaks, parent_mass):
        return n_peaks / parent_mass if parent_mass > 0 else 0.0

    @staticmethod
    def intensity_stats(intensities):
        if not intensities:
            return (0.0, 0.0)
        mean = sum(intensities) / len(intensities)
        var = sum((x - mean) ** 2 for x in intensities) / len(intensities)
        return mean, math.sqrt(var)

    @staticmethod
    def highmass_fraction(mz, intensities, parent_mass):
        if parent_mass <= 0:
            return 0.0
        cutoff = 0.5 * parent_mass
        total = sum(intensities)
        if total == 0:
            return 0.0
        return sum(i for m, i in zip(mz, intensities) if m >= cutoff) / total

    @staticmethod
    def lowmass_fraction(mz, intensities, parent_mass):
        if parent_mass <= 0:
            return 0.0
        cutoff = 0.2 * parent_mass
        total = sum(intensities)
        if total == 0:
            return 0.0
        return sum(i for m, i in zip(mz, intensities) if m <= cutoff) / total

    # --------------------------------------------------------
    # Aromatic / halogen pattern detectors
    # --------------------------------------------------------
    @staticmethod
    def has_peak(mz_list, target, tol=0.5):
        return any(abs(m - target) <= tol for m in mz_list)

    @staticmethod
    def detect_cl_pattern(mz, intensities):
        # ~3:1 Cl isotope pattern
        for m, i in zip(mz, intensities):
            m2 = m + 2
            for m2p, i2 in zip(mz, intensities):
                if abs(m2p - m2) <= 0.3 and i > 0 and 0.2 < (i2 / i) < 0.4:
                    return 1
        return 0

    @staticmethod
    def detect_br_pattern(mz, intensities):
        # ~1:1 Br isotope pattern
        for m, i in zip(mz, intensities):
            m2 = m + 2
            for m2p, i2 in zip(mz, intensities):
                if abs(m2p - m2) <= 0.3 and i > 0 and 0.8 < (i2 / i) < 1.2:
                    return 1
        return 0

    # --------------------------------------------------------
    # Local peak context
    # --------------------------------------------------------
    @staticmethod
    def local_intensity(mz_list, int_list, target, tol=0.5):
        for m, i in zip(mz_list, int_list):
            if abs(m - target) <= tol:
                return i
        return 0.0

    @staticmethod
    def local_peak_density(mz_list, target, window=5):
        return sum(1 for m in mz_list if abs(m - target) <= window)

    # --------------------------------------------------------
    # Build feature row for a single assignment
    # --------------------------------------------------------
    def build_feature_row(
        self,
        parent_name: str,
        parent_formula_str: str,
        nist_mz,
        nist_int,
        assignment,
    ) -> pd.DataFrame:
        """
        Build a 1-row DataFrame with the same features used in training.

        Parameters
        ----------
        parent_name : str
        parent_formula_str : str
        nist_mz : list[float]
        nist_int : list[float]
        assignment : object
            Must have attributes:
            - best_formula (str)
            - best_exact_mz (float)
            - nominal_mz (int)
            - intensity (float)
            - confidence (float)
            - rule_source (str or None)
        """

        # parent formula
        parent_formula = Formula.from_string(parent_formula_str)
        parent_mass = exact_mass(parent_formula)
        parent_dbe = dbe(parent_formula)
        elems = parent_formula.elements
        n_C = elems.get("C", 0)
        n_H = elems.get("H", 0)
        n_O = elems.get("O", 0)
        n_N = elems.get("N", 0)
        n_halogen = sum(elems.get(x, 0) for x in ["Cl", "Br", "F", "I"])

        # classes
        classes_set = classify_molecule(parent_name, parent_formula_str)

        # NIST global descriptors
        nist_n_peaks = len(nist_mz)
        base_idx = max(range(len(nist_int)), key=lambda i: nist_int[i])
        nist_base_mz = nist_mz[base_idx]
        nist_base_intensity = nist_int[base_idx]

        ent = self.nist_entropy(nist_int)
        pdens = self.peak_density(nist_n_peaks, parent_mass)
        mean_int, std_int = self.intensity_stats(nist_int)
        high_frac = self.highmass_fraction(nist_mz, nist_int, parent_mass)
        low_frac = self.lowmass_fraction(nist_mz, nist_int, parent_mass)

        aromatic_77 = int(self.has_peak(nist_mz, 77))
        aromatic_91 = int(self.has_peak(nist_mz, 91))
        aromatic_105 = int(self.has_peak(nist_mz, 105))

        cl_pat = self.detect_cl_pattern(nist_mz, nist_int)
        br_pat = self.detect_br_pattern(nist_mz, nist_int)

        # relative intensities
        max_int = max(nist_int)
        rel_int = [i / max_int for i in nist_int]

        # filtered peaks (for rel_int lookup)
        nist_peaks_all = list(zip(nist_mz, nist_int, rel_int))
        nist_peaks_filtered = [
            (mz, I, rI) for (mz, I, rI) in nist_peaks_all
            if rI >= 0.05
        ]
        rel_int_by_nominal = {mz: rI for (mz, _, rI) in nist_peaks_filtered}

        # fragment-level
        frag_formula_str = assignment.best_formula
        frag_mass = assignment.best_exact_mz
        try:
            frag_formula = Formula.from_string(frag_formula_str)
            frag_dbe = dbe(frag_formula)
        except Exception:
            frag_dbe = 0.0

        mass_fraction = frag_mass / parent_mass if parent_mass > 0 else 0.0

        rule_source = assignment.rule_source or ""
        if " (depth=" in rule_source:
            rule_family = rule_source.split(" (depth=")[0]
        else:
            rule_family = rule_source or "none"

        peak_nominal_mz = assignment.nominal_mz
        peak_intensity = assignment.intensity
        peak_rel_intensity = rel_int_by_nominal.get(peak_nominal_mz, 0.0)

        # local context
        local_i = self.local_intensity(nist_mz, nist_int, peak_nominal_mz)
        local_i_m1 = self.local_intensity(nist_mz, nist_int, peak_nominal_mz - 1)
        local_i_p1 = self.local_intensity(nist_mz, nist_int, peak_nominal_mz + 1)
        local_i_m14 = self.local_intensity(nist_mz, nist_int, peak_nominal_mz - 14)
        local_i_p14 = self.local_intensity(nist_mz, nist_int, peak_nominal_mz + 14)
        local_density = self.local_peak_density(nist_mz, peak_nominal_mz)

        # base feature dict (numeric + rule_family)
        row = {
            "parent_mass": parent_mass,
            "parent_dbe": parent_dbe,
            "n_C": n_C,
            "n_H": n_H,
            "n_O": n_O,
            "n_N": n_N,
            "n_halogen": n_halogen,

            "nist_n_peaks": nist_n_peaks,
            "nist_base_mz": nist_base_mz,
            "nist_base_intensity": nist_base_intensity,
            "nist_entropy": ent,
            "nist_peak_density": pdens,
            "nist_intensity_mean": mean_int,
            "nist_intensity_std": std_int,
            "nist_highmass_fraction": high_frac,
            "nist_lowmass_fraction": low_frac,
            "has_peak_77": aromatic_77,
            "has_peak_91": aromatic_91,
            "has_peak_105": aromatic_105,
            "has_cl_pattern": cl_pat,
            "has_br_pattern": br_pat,

            "peak_nominal_mz": peak_nominal_mz,
            "peak_intensity": peak_intensity,
            "peak_rel_intensity": peak_rel_intensity,
            "local_intensity_mz": local_i,
            "local_intensity_mz_minus1": local_i_m1,
            "local_intensity_mz_plus1": local_i_p1,
            "local_intensity_mz_minus14": local_i_m14,
            "local_intensity_mz_plus14": local_i_p14,
            "local_peak_density": local_density,

            "frag_mass": frag_mass,
            "frag_dbe": frag_dbe,
            "mass_fraction": mass_fraction,
            "confidence": assignment.confidence,
            "rule_family": rule_family,
        }

        # class_* features: set 1 for present classes, 0 otherwise
        for fname in self.feature_names:
            if fname.startswith("class_"):
                cls = fname[len("class_"):]
                row[fname] = 1 if cls in classes_set else 0

        # ensure all expected features exist
        for fname in self.feature_names:
            if fname not in row:
                # unseen numeric feature → 0, unseen categorical handled by LightGBM
                row[fname] = 0

        # build DataFrame with correct column order
        df_row = pd.DataFrame([row], columns=self.feature_names)

        # rule_family as category
        if "rule_family" in df_row.columns:
            df_row["rule_family"] = df_row["rule_family"].astype("category")

        return df_row

    # --------------------------------------------------------
    # Predict probability for a single assignment
    # --------------------------------------------------------
    def predict_prob(
        self,
        parent_name: str,
        parent_formula_str: str,
        nist_mz,
        nist_int,
        assignment,
    ) -> float:
        features = self.build_feature_row(
            parent_name=parent_name,
            parent_formula_str=parent_formula_str,
            nist_mz=nist_mz,
            nist_int=nist_int,
            assignment=assignment,
        )
        # Booster.predict with pandas aligns by column name
        prob = self.booster.predict(features)[0]
        return float(prob)