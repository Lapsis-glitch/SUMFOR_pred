# collect_training_data_v2.py
#
# Regenerate ML training data using the current fragmentation pipeline.
#
# Uses the same fragment generation as Large_data.run_single_entry()
# (HybridEnumerator + PeakDrivenAssignmentEngine) to ensure training
# data matches inference. Labels are assigned by matching predicted
# fragment exact masses against AML high-resolution reference peaks.
#
# Parallelised with multiprocessing for speed.
#
# Usage:
#   cd src/Nist_HR
#   python collect_training_data_v2.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
import math
import multiprocessing as mp

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tqdm import tqdm

from formula import Formula
from chemistry import exact_mass, dbe
from peak_driven_enumerator import PeakDrivenEnumerator
from peak_driven_assignment_engine import PeakDrivenAssignmentEngine
from hybrid_enumerator import HybridEnumerator
from chemical_classification import classify_molecule

# ── Configuration ─────────────────────────────────────────────
# These MUST match Large_data.py so training = inference.
MERGED_PATH = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"
OUT_CSV = "training_fragments_SIP_hybrid_BDE_v2.csv"

MIN_REL_INTENSITY = 0.10
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "Cl", "Br", "F", "I", "S", "P"}
FRAG_DEPTH = 5
BDE_THRESHOLD = 120.0
BDE_SOFTNESS = 50.0
MASS_TOL = 0.0002       # same as validation TOL

RULE_FLAGS = {
    "alcohol": True,
    "carbonyl": True,
    "aromatic": True,
    "amine": True,
    "ester": True,
    "halogen": True,
    "ether": True,
    "alkene": True,
    "sulfur": True,
    "phosphorus": True,
}

N_WORKERS = min(os.cpu_count() or 1, 14)


# ── Helpers (same as ml_correction.py / collect_training_data_bde.py) ──

def is_correct(frag_mass, aml_mz_list, tol=MASS_TOL):
    return any(abs(frag_mass - m) <= tol for m in aml_mz_list)


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
        return 0.0, 0.0
    mean = sum(intensities) / len(intensities)
    var = sum((x - mean) ** 2 for x in intensities) / len(intensities)
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


def has_peak(mz_list, target, tol=0.5):
    return any(abs(m - target) <= tol for m in mz_list)


def detect_cl_pattern(mz, intensities):
    for m, i in zip(mz, intensities):
        m2 = m + 2
        for m2p, i2 in zip(mz, intensities):
            if abs(m2p - m2) <= 0.3 and i > 0 and 0.2 < (i2 / i) < 0.4:
                return 1
    return 0


def detect_br_pattern(mz, intensities):
    for m, i in zip(mz, intensities):
        m2 = m + 2
        for m2p, i2 in zip(mz, intensities):
            if abs(m2p - m2) <= 0.3 and i > 0 and 0.8 < (i2 / i) < 1.2:
                return 1
    return 0


def detect_i_pattern(mz, intensities):
    for m, i in zip(mz, intensities):
        if abs(m - 127.0) <= 0.3 and i > 0:
            return 1
    return 0


def local_intensity(mz_list, int_list, target, tol=0.5):
    for m, i in zip(mz_list, int_list):
        if abs(m - target) <= tol:
            return i
    return 0.0


def local_peak_density(mz_list, target, window=5):
    return sum(1 for m in mz_list if abs(m - target) <= window)


def _build_mol(entry):
    pubchem = entry.get("pubchem")
    if not pubchem:
        return None
    smiles = pubchem.get("canonical_smiles") or pubchem.get("isomeric_smiles")
    inchi = pubchem.get("inchi")
    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    elif inchi:
        mol = Chem.MolFromInchi(inchi)
    if mol is not None:
        mol = Chem.AddHs(mol)
    return mol


def _extract_bde_data(entry):
    raw_bde = entry.get("bde_data")
    if not raw_bde or "bonds" not in raw_bde:
        return None
    bonds = [b for b in raw_bde["bonds"] if b["bde"] is not None]
    return bonds if bonds else None


# ── CSV field names ───────────────────────────────────────────

OUT_FIELDS = [
    "entry_id",
    "parent_name", "parent_formula", "parent_mass", "parent_dbe",
    "n_C", "n_H", "n_O", "n_N", "n_S", "n_P",
    "n_F", "n_Cl", "n_Br", "n_I", "n_halogen",
    "classes",
    "nist_n_peaks", "nist_base_mz", "nist_base_intensity",
    "nist_entropy", "nist_peak_density",
    "nist_intensity_mean", "nist_intensity_std",
    "nist_highmass_fraction", "nist_lowmass_fraction",
    "has_peak_77", "has_peak_91", "has_peak_105",
    "has_cl_pattern", "has_br_pattern", "has_i_pattern",
    "peak_nominal_mz", "peak_intensity", "peak_rel_intensity",
    "local_intensity_mz", "local_intensity_mz_minus1",
    "local_intensity_mz_plus1", "local_intensity_mz_minus14",
    "local_intensity_mz_plus14", "local_peak_density",
    "frag_formula", "frag_mass", "frag_dbe",
    "mass_fraction", "confidence", "rule_family",
    "frag_n_C", "frag_n_H", "frag_n_O", "frag_n_N",
    "frag_n_S", "frag_n_P", "frag_n_F", "frag_n_Cl",
    "frag_n_Br", "frag_n_I", "frag_n_halogen",
    # BDE / pathway features
    "is_bde_fragment",
    "n_bde_steps", "n_rule_steps", "path_length",
    "contains_aromatic_rule", "contains_neutral_loss", "contains_rearrangement",
    # Fragment structural features
    "mass_defect", "H_to_C_ratio", "N_to_C_ratio", "O_to_C_ratio",
    "is_common_ei_ion",
    # Peak proximity
    "distance_to_nearest_peak", "intensity_of_nearest_peak",
    "within_1Da", "within_2Da",
    # Label
    "label",
]


# ── Process one entry (runs in worker process) ───────────────

def process_entry(entry_id):
    """
    Run the full pipeline for one entry, compute features + labels,
    return a list of row dicts (one per fragment).
    """
    entry = MERGED[entry_id]
    csv_entry = entry["csv_entry"]
    aml_mz = csv_entry.get("mz", [])
    aml_int = csv_entry.get("intensities", [])
    parent_name = csv_entry.get("name", "")

    if not aml_mz or not aml_int:
        return []

    if not entry.get("nist_matches"):
        return []

    nist = entry["nist_matches"][0]
    nist_mz = nist.get("mz", [])
    nist_int = nist.get("intensities", [])
    if not nist_mz or not nist_int:
        return []

    parent_formula_str = nist.get("sum_formula") or csv_entry.get("sum_formula")
    if not parent_formula_str:
        return []

    try:
        parent_formula = Formula.from_string(parent_formula_str)
    except Exception:
        return []

    if not all(el in ALLOWED_ELEMENTS for el in parent_formula.elements):
        return []

    parent_mass = exact_mass(parent_formula)
    parent_dbe = dbe(parent_formula)
    elems = parent_formula.elements

    n_C  = elems.get("C", 0)
    n_H  = elems.get("H", 0)
    n_O  = elems.get("O", 0)
    n_N  = elems.get("N", 0)
    n_S  = elems.get("S", 0)
    n_P  = elems.get("P", 0)
    n_F  = elems.get("F", 0)
    n_Cl = elems.get("Cl", 0)
    n_Br = elems.get("Br", 0)
    n_I  = elems.get("I", 0)
    n_halogen = n_F + n_Cl + n_Br + n_I

    classes_set = set(classify_molecule(parent_name, parent_formula_str))
    if n_S > 0:  classes_set.add("sulfur")
    if n_P > 0:  classes_set.add("phosphorus")
    if n_F > 0:  classes_set.add("fluorinated")
    if n_Cl > 0: classes_set.add("chlorinated")
    if n_Br > 0: classes_set.add("brominated")
    if n_I > 0:  classes_set.add("iodinated")
    classes_str = ";".join(sorted(classes_set))

    # ── NIST global descriptors ──
    nist_n_peaks = len(nist_mz)
    base_idx = max(range(len(nist_int)), key=lambda i: nist_int[i])
    nist_base_mz_val = nist_mz[base_idx]
    nist_base_int_val = nist_int[base_idx]

    ent = nist_entropy(nist_int)
    pdens = peak_density(nist_n_peaks, parent_mass)
    mean_int, std_int = intensity_stats(nist_int)
    high_frac = highmass_fraction(nist_mz, nist_int, parent_mass)
    low_frac = lowmass_fraction(nist_mz, nist_int, parent_mass)

    aromatic_77  = int(has_peak(nist_mz, 77))
    aromatic_91  = int(has_peak(nist_mz, 91))
    aromatic_105 = int(has_peak(nist_mz, 105))

    cl_pat = detect_cl_pattern(nist_mz, nist_int)
    br_pat = detect_br_pattern(nist_mz, nist_int)
    i_pat  = detect_i_pattern(nist_mz, nist_int)

    max_int = max(nist_int)
    rel_int = [i / max_int for i in nist_int]

    nist_peaks_all = list(zip(nist_mz, nist_int, rel_int))
    nist_peaks_filtered = [
        (mz, I, rI) for mz, I, rI in nist_peaks_all
        if rI >= MIN_REL_INTENSITY
    ]
    if not nist_peaks_filtered:
        return []

    mz_filt  = [mz for mz, _, _ in nist_peaks_filtered]
    int_filt = [I  for _, I, _ in nist_peaks_filtered]
    rel_filt = [rI for _, _, rI in nist_peaks_filtered]
    rel_int_by_nominal = {mz: rI for mz, _, rI in nist_peaks_filtered}

    # ── Build enumerator (same logic as Large_data.run_single_entry) ──
    mol = _build_mol(entry)
    bde_data = _extract_bde_data(entry)

    if mol is not None and bde_data:
        peak_enum = HybridEnumerator(
            parent_formula, mol, bde_data,
            rule_flags=RULE_FLAGS,
            max_depth=FRAG_DEPTH,
            bde_threshold=BDE_THRESHOLD,
            bde_softness=BDE_SOFTNESS,
        )
    else:
        peak_enum = PeakDrivenEnumerator(
            parent_formula,
            rule_flags=RULE_FLAGS,
            max_depth=FRAG_DEPTH,
            auto_detect_rules=True,
        )

    # ── Assign peaks ──
    engine = PeakDrivenAssignmentEngine(parent_formula, peak_enum)
    assignments = engine.assign_peaks(
        nist_peaks=list(zip(mz_filt, int_filt)),
        rel_intensities=rel_filt,
    )

    # ── Build rows ──
    rows = []
    for a in assignments:
        if a.best_formula is None or a.best_exact_mz is None:
            continue

        frag_formula_str = a.best_formula
        frag_mass = a.best_exact_mz

        try:
            frag_formula = Formula.from_string(frag_formula_str)
            frag_dbe_val = dbe(frag_formula)
        except Exception:
            frag_dbe_val = 0.0
            frag_formula = Formula({})

        fe = frag_formula.elements
        frag_n_C  = fe.get("C", 0)
        frag_n_H  = fe.get("H", 0)
        frag_n_O  = fe.get("O", 0)
        frag_n_N  = fe.get("N", 0)
        frag_n_S  = fe.get("S", 0)
        frag_n_P  = fe.get("P", 0)
        frag_n_F  = fe.get("F", 0)
        frag_n_Cl = fe.get("Cl", 0)
        frag_n_Br = fe.get("Br", 0)
        frag_n_I  = fe.get("I", 0)
        frag_n_halogen = frag_n_F + frag_n_Cl + frag_n_Br + frag_n_I

        mass_fraction = frag_mass / parent_mass if parent_mass > 0 else 0.0

        rule_source = a.rule_source or ""
        rule_family = rule_source.split("_")[0] if rule_source else "none"

        nominal = a.nominal_mz
        loc_mz    = local_intensity(nist_mz, nist_int, nominal)
        loc_mz_m1 = local_intensity(nist_mz, nist_int, nominal - 1)
        loc_mz_p1 = local_intensity(nist_mz, nist_int, nominal + 1)
        loc_mz_m14 = local_intensity(nist_mz, nist_int, nominal - 14)
        loc_mz_p14 = local_intensity(nist_mz, nist_int, nominal + 14)
        loc_density = local_peak_density(nist_mz, nominal)

        label = int(is_correct(frag_mass, aml_mz))

        # ── BDE / pathway features ──
        rs_lower = rule_source.lower()
        is_bde_fragment = int("bde" in rs_lower)
        n_bde_steps = rs_lower.count("bde")
        parts = rule_source.split("+") if rule_source else []
        n_rule_steps = len(parts) - n_bde_steps
        path_length = len(parts)

        contains_aromatic_rule = int("aromatic" in rs_lower)
        contains_neutral_loss = int("neutral" in rs_lower)
        contains_rearrangement = int("rearrang" in rs_lower)

        # ── Fragment structural features ──
        mass_defect = frag_mass - round(frag_mass)
        H_to_C = frag_n_H / frag_n_C if frag_n_C > 0 else 0.0
        N_to_C = frag_n_N / frag_n_C if frag_n_C > 0 else 0.0
        O_to_C = frag_n_O / frag_n_C if frag_n_C > 0 else 0.0
        is_common_ei = int(round(frag_mass) in {39, 51, 65, 77, 91, 105})

        # ── Peak proximity ──
        distances = [abs(m - frag_mass) for m in nist_mz]
        dist_nearest = min(distances) if distances else 999.0
        if distances:
            nearest_idx = distances.index(dist_nearest)
            int_nearest = nist_int[nearest_idx]
        else:
            int_nearest = 0.0
        within_1 = int(dist_nearest <= 1.0)
        within_2 = int(dist_nearest <= 2.0)

        rows.append({
            "entry_id": entry_id,
            "parent_name": parent_name,
            "parent_formula": parent_formula_str,
            "parent_mass": parent_mass,
            "parent_dbe": parent_dbe,
            "n_C": n_C, "n_H": n_H, "n_O": n_O, "n_N": n_N,
            "n_S": n_S, "n_P": n_P,
            "n_F": n_F, "n_Cl": n_Cl, "n_Br": n_Br, "n_I": n_I,
            "n_halogen": n_halogen,
            "classes": classes_str,
            "nist_n_peaks": nist_n_peaks,
            "nist_base_mz": nist_base_mz_val,
            "nist_base_intensity": nist_base_int_val,
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
            "has_i_pattern": i_pat,
            "peak_nominal_mz": nominal,
            "peak_intensity": loc_mz,
            "peak_rel_intensity": rel_int_by_nominal.get(nominal, 0.0),
            "local_intensity_mz": loc_mz,
            "local_intensity_mz_minus1": loc_mz_m1,
            "local_intensity_mz_plus1": loc_mz_p1,
            "local_intensity_mz_minus14": loc_mz_m14,
            "local_intensity_mz_plus14": loc_mz_p14,
            "local_peak_density": loc_density,
            "frag_formula": frag_formula_str,
            "frag_mass": frag_mass,
            "frag_dbe": frag_dbe_val,
            "mass_fraction": mass_fraction,
            "confidence": a.confidence,
            "rule_family": rule_family,
            "frag_n_C": frag_n_C, "frag_n_H": frag_n_H,
            "frag_n_O": frag_n_O, "frag_n_N": frag_n_N,
            "frag_n_S": frag_n_S, "frag_n_P": frag_n_P,
            "frag_n_F": frag_n_F, "frag_n_Cl": frag_n_Cl,
            "frag_n_Br": frag_n_Br, "frag_n_I": frag_n_I,
            "frag_n_halogen": frag_n_halogen,
            "is_bde_fragment": is_bde_fragment,
            "n_bde_steps": n_bde_steps,
            "n_rule_steps": n_rule_steps,
            "path_length": path_length,
            "contains_aromatic_rule": contains_aromatic_rule,
            "contains_neutral_loss": contains_neutral_loss,
            "contains_rearrangement": contains_rearrangement,
            "mass_defect": mass_defect,
            "H_to_C_ratio": H_to_C,
            "N_to_C_ratio": N_to_C,
            "O_to_C_ratio": O_to_C,
            "is_common_ei_ion": is_common_ei,
            "distance_to_nearest_peak": dist_nearest,
            "intensity_of_nearest_peak": int_nearest,
            "within_1Da": within_1,
            "within_2Da": within_2,
            "label": label,
        })

    return rows


# ── Worker initialiser ────────────────────────────────────────

def _init_worker():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


# ── Main ──────────────────────────────────────────────────────

MERGED = None  # populated in main / inherited via fork

def main():
    global MERGED

    print(f"Loading dataset from {MERGED_PATH} ...")
    with open(MERGED_PATH, "r") as f:
        MERGED = json.load(f)

    all_ids = sorted(MERGED.keys(), key=lambda x: int(x))

    from heldout_split import load_heldout_test_ids
    heldout_ids = load_heldout_test_ids()
    if heldout_ids:
        entry_ids = [eid for eid in all_ids if str(eid) not in heldout_ids]
        print(
            f"  {len(entry_ids)} entries (excluded {len(all_ids) - len(entry_ids)} "
            f"held-out test entries out of {len(heldout_ids)} reserved)"
        )
    else:
        entry_ids = all_ids
        print(f"  {len(entry_ids)} entries (no heldout_test_entries.json — using full corpus)")

    print(f"  Workers: {N_WORKERS}")
    print(f"  Output:  {OUT_CSV}")
    print(f"  Config:  FRAG_DEPTH={FRAG_DEPTH}  MIN_REL={MIN_REL_INTENSITY}  "
          f"BDE_THR={BDE_THRESHOLD}  BDE_SOFT={BDE_SOFTNESS}  TOL={MASS_TOL}")

    total_rows = 0
    total_pos = 0
    entries_ok = 0

    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUT_FIELDS)
        writer.writeheader()

        with mp.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
            results_iter = pool.imap_unordered(process_entry, entry_ids)

            for rows in tqdm(results_iter, total=len(entry_ids),
                             desc="Collecting", unit="entry"):
                if not rows:
                    continue
                entries_ok += 1
                for row in rows:
                    writer.writerow(row)
                    total_rows += 1
                    if row["label"] == 1:
                        total_pos += 1

    print(f"\nDone.")
    print(f"  Entries processed: {entries_ok}/{len(entry_ids)}")
    print(f"  Total fragments:   {total_rows}")
    print(f"  Positive (label=1): {total_pos} ({100*total_pos/max(total_rows,1):.1f}%)")
    print(f"  Negative (label=0): {total_rows - total_pos}")
    print(f"  Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()

