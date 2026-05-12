# utils.py

from __future__ import annotations
import csv
from typing import List, Tuple

def convert_assignments(assignments):
    converted = []
    for a in assignments:
        converted.append({
            "nominal_mz": a.nominal_mz,
            "intensity": a.intensity,
            "best_exact_mz": a.best_exact_mz,
            "best_formula": a.best_formula,
            "candidate_formulas": a.candidate_formulas,
            "conf": a.confidence,
            "score": a.confidence,
            "ml_prob": getattr(a, "ml_prob", 0.0),  # ✔ propagate ML score
            "rule_source": a.rule_source,
        })
    return converted




def write_assignments_csv(assignments, filename):
    """
    Write assignment dictionaries to CSV.
    Compatible with multi-match mode (one row per fragment candidate).
    """
    import csv

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "nominal_mz",
            "intensity",
            "best_exact_mz",
            "best_formula",
            "candidate_formulas",
            "confidence",
            "rule_source",
        ])

        # Rows
        for a in assignments:
            writer.writerow([
                a["nominal_mz"],
                a["intensity"],
                a["best_exact_mz"],
                a["best_formula"],
                ";".join(a["candidate_formulas"]),  # safe for CSV
                a["conf"],
                a["rule_source"],
            ])


def nist_bin(mass: float) -> int | None:
    """
    Asymmetric NIST window: assign to integer m if mass ∈ [m-0.3, m+0.7).
    """
    m_int = round(mass)
    if (mass >= m_int - 0.3) and (mass < m_int + 0.7):
        return m_int
    return None


def parse_reference_csv(filepath: str) -> List[Tuple[float, float]]:
    """
    Parse reference CSV with columns:
    Spectrum,TOF,M/Z,Area,Resolution,ClusterId

    Returns list of (mz, intensity)
    """
    peaks = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mz = float(row["M/Z"])
                inten = float(row["Area"])
                peaks.append((mz, inten))
            except (ValueError, KeyError):
                continue
    return peaks