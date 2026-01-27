# collect_training_data.py

import json
import csv
import math
from pathlib import Path

from formula import Formula
from chemistry import exact_mass, dbe
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from utils import convert_assignments
from chemical_classification import classify_molecule

MIN_REL_INTENSITY = 0.05
MERGED_PATH = "/mnt/d/Leco/merged_clean.json"
OUT_CSV     = "training_fragments.csv"

MASS_TOL = 0.0001


# ------------------------------------------------------------
# Helper: label correctness by matching to AML peaks
# ------------------------------------------------------------
def is_correct_assignment(frag_mass, ref_peaks, tol=MASS_TOL):
    for mz, _ in ref_peaks:
        if abs(mz - frag_mass) <= tol:
            return True
    return False


# ------------------------------------------------------------
# NIST global descriptors
# ------------------------------------------------------------
def nist_entropy(intensities):
    total = sum(intensities)
    if total == 0:
        return 0.0
    p = [i / total for i in intensities]
    return -sum(pi * math.log(pi) for pi in p if pi > 0)


def peak_density(n_peaks, parent_mass):
    return n_peaks / parent_mass if parent_mass > 0 else 0.0


def intensity_stats(intensities):
    if not intensities:
        return (0.0, 0.0)
    mean = sum(intensities) / len(intensities)
    var = sum((x - mean)**2 for x in intensities) / len(intensities)
    return mean, math.sqrt(var)


def highmass_fraction(mz, intensities, parent_mass):
    if parent_mass <= 0:
        return 0.0
    cutoff = 0.5 * parent_mass
    total = sum(intensities)
    if total == 0:
        return 0.0
    return sum(i for m, i in zip(mz, intensities) if m >= cutoff) / total


def lowmass_fraction(mz, intensities, parent_mass):
    if parent_mass <= 0:
        return 0.0
    cutoff = 0.2 * parent_mass
    total = sum(intensities)
    if total == 0:
        return 0.0
    return sum(i for m, i in zip(mz, intensities) if m <= cutoff) / total


# ------------------------------------------------------------
# Aromatic / halogen pattern detectors
# ------------------------------------------------------------
def has_peak(mz_list, target, tol=0.5):
    return any(abs(m - target) <= tol for m in mz_list)


def detect_cl_pattern(mz, intensities):
    # Look for ~3:1 Cl isotope pattern
    for m, i in zip(mz, intensities):
        m2 = m + 2
        for m2p, i2 in zip(mz, intensities):
            if abs(m2p - m2) <= 0.3 and i > 0 and 0.2 < (i2 / i) < 0.4:
                return 1
    return 0


def detect_br_pattern(mz, intensities):
    # Look for ~1:1 Br isotope pattern
    for m, i in zip(mz, intensities):
        m2 = m + 2
        for m2p, i2 in zip(mz, intensities):
            if abs(m2p - m2) <= 0.3 and i > 0 and 0.8 < (i2 / i) < 1.2:
                return 1
    return 0


# ------------------------------------------------------------
# Local peak context around nominal m/z
# ------------------------------------------------------------
def local_intensity(mz_list, int_list, target, tol=0.5):
    for m, i in zip(mz_list, int_list):
        if abs(m - target) <= tol:
            return i
    return 0.0


def local_peak_density(mz_list, target, window=5):
    return sum(1 for m in mz_list if abs(m - target) <= window)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    with open(MERGED_PATH, "r") as f:
        merged = json.load(f)

    entry_ids = sorted(merged.keys(), key=lambda x: int(x))

    out_fields = [
        "entry_id",

        # parent / molecule
        "parent_name",
        "parent_formula",
        "parent_mass",
        "parent_dbe",
        "n_C",
        "n_H",
        "n_O",
        "n_N",
        "n_halogen",
        "classes",

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

        # peak-level
        "peak_nominal_mz",
        "peak_intensity",
        "peak_rel_intensity",
        "local_intensity_mz",
        "local_intensity_mz_minus1",
        "local_intensity_mz_plus1",
        "local_intensity_mz_minus14",
        "local_intensity_mz_plus14",
        "local_peak_density",

        # fragment-level
        "frag_formula",
        "frag_mass",
        "frag_dbe",
        "mass_fraction",
        "confidence",
        "rule_family",

        # label
        "label",
    ]

    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for entry_id in entry_ids:
            entry = merged[entry_id]

            csv_entry = entry["csv_entry"]
            aml_mz = csv_entry["mz"]
            aml_int = csv_entry["intensities"]
            parent_name = csv_entry.get("name", "")

            if not aml_mz or not aml_int:
                continue

            # NIST: take first match
            if not entry["nist_matches"]:
                continue

            nist = entry["nist_matches"][0]
            nist_mz = nist["mz"]
            nist_int = nist["intensities"]

            if not nist_mz or not nist_int:
                continue

            # parent formula
            parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
            if not parent_formula_str:
                continue

            try:
                parent_formula = Formula.from_string(parent_formula_str)
            except Exception:
                continue

            parent_mass = exact_mass(parent_formula)
            parent_dbe  = dbe(parent_formula)
            elems = parent_formula.elements
            n_C = elems.get("C", 0)
            n_H = elems.get("H", 0)
            n_O = elems.get("O", 0)
            n_N = elems.get("N", 0)
            n_halogen = sum(elems.get(x, 0) for x in ["Cl", "Br", "F", "I"])

            # chemical classes
            classes_set = classify_molecule(parent_name, parent_formula_str)
            classes_str = ";".join(sorted(classes_set))

            # NIST global descriptors
            nist_n_peaks = len(nist_mz)
            base_idx = max(range(len(nist_int)), key=lambda i: nist_int[i])
            nist_base_mz = nist_mz[base_idx]
            nist_base_intensity = nist_int[base_idx]

            ent = nist_entropy(nist_int)
            pdens = peak_density(nist_n_peaks, parent_mass)
            mean_int, std_int = intensity_stats(nist_int)
            high_frac = highmass_fraction(nist_mz, nist_int, parent_mass)
            low_frac = lowmass_fraction(nist_mz, nist_int, parent_mass)

            aromatic_77 = int(has_peak(nist_mz, 77))
            aromatic_91 = int(has_peak(nist_mz, 91))
            aromatic_105 = int(has_peak(nist_mz, 105))

            cl_pat = detect_cl_pattern(nist_mz, nist_int)
            br_pat = detect_br_pattern(nist_mz, nist_int)

            # relative intensities
            max_int = max(nist_int)
            rel_int = [i / max_int for i in nist_int]

            # filter NIST peaks
            nist_peaks_all = list(zip(nist_mz, nist_int, rel_int))
            nist_peaks_filtered = [
                (mz, I, rI) for (mz, I, rI) in nist_peaks_all
                if rI >= MIN_REL_INTENSITY
            ]

            if not nist_peaks_filtered:
                continue

            mz_ref_filt = [mz for mz, _, _ in nist_peaks_filtered]
            intensity_ref_filt = [I for _, I, _ in nist_peaks_filtered]
            rel_int_filt = [rI for _, _, rI in nist_peaks_filtered]

            # fragment rules
            FRAG_DEPTH = 3
            rule_flags = {
                "neutral_losses": True,
                "double_neutral_losses": True,
                "common_cations": True,
                "alpha_cleavage": True,
                "rearrangements": True,
                "oxygen_adjacent": True,
                "hydrogen_transfer": True,
                "mclafferty": True,

                "alcohol_rules": True,
                "carbonyl_rules": True,
                "aromatic_rules": True,
                "amine_rules": True,
                "ester_rules": True,
                "halogen_rules": True,
                "ether_rules": True,
                "alkene_rules": True,
            }

            peak_enum = PeakDrivenEnumerator(
                parent_formula,
                rule_flags=rule_flags,
                max_depth=FRAG_DEPTH,
                auto_detect_rules=True,
            )

            engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)

            nist_peaks = list(zip(mz_ref_filt, intensity_ref_filt))
            assignments = engine.assign_peaks(
                nist_peaks=nist_peaks,
                rel_intensities=rel_int_filt,
            )

            # AML reference
            ref_peaks = list(zip(aml_mz, aml_int))

            # rel intensity lookup
            rel_int_by_nominal = {mz: rI for (mz, _, rI) in nist_peaks_filtered}

            for a in assignments:
                if a.best_formula is None or a.best_exact_mz is None:
                    continue

                frag_formula_str = a.best_formula
                frag_mass = a.best_exact_mz

                try:
                    frag_formula = Formula.from_string(frag_formula_str)
                    frag_dbe = dbe(frag_formula)
                except Exception:
                    frag_dbe = 0.0

                mass_fraction = frag_mass / parent_mass if parent_mass > 0 else 0.0

                rule_source = a.rule_source or ""
                if " (depth=" in rule_source:
                    rule_family = rule_source.split(" (depth=")[0]
                else:
                    rule_family = rule_source

                peak_nominal_mz = a.nominal_mz
                peak_intensity = a.intensity
                peak_rel_intensity = rel_int_by_nominal.get(peak_nominal_mz, 0.0)

                # local context
                local_i = local_intensity(nist_mz, nist_int, peak_nominal_mz)
                local_i_m1 = local_intensity(nist_mz, nist_int, peak_nominal_mz - 1)
                local_i_p1 = local_intensity(nist_mz, nist_int, peak_nominal_mz + 1)
                local_i_m14 = local_intensity(nist_mz, nist_int, peak_nominal_mz - 14)
                local_i_p14 = local_intensity(nist_mz, nist_int, peak_nominal_mz + 14)
                local_density = local_peak_density(nist_mz, peak_nominal_mz)

                label = 1 if is_correct_assignment(frag_mass, ref_peaks) else 0

                row = {
                    "entry_id": entry_id,

                    "parent_name": parent_name,
                    "parent_formula": parent_formula_str,
                    "parent_mass": parent_mass,
                    "parent_dbe": parent_dbe,
                    "n_C": n_C,
                    "n_H": n_H,
                    "n_O": n_O,
                    "n_N": n_N,
                    "n_halogen": n_halogen,
                    "classes": classes_str,

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

                    "frag_formula": frag_formula_str,
                    "frag_mass": frag_mass,
                    "frag_dbe": frag_dbe,
                    "mass_fraction": mass_fraction,
                    "confidence": a.confidence,
                    "rule_family": rule_family,

                    "label": label,
                }

                writer.writerow(row)

    print(f"Wrote training data to {OUT_CSV}")


if __name__ == "__main__":
    main()