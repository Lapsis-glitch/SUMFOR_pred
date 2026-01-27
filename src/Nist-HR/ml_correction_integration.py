# ml_correction_integration.py

from ml_correction import MLCorrectionModel

# Load once globally
ML_MODEL = MLCorrectionModel()

def apply_ml_correction(
    assignments,
    parent_name,
    parent_formula_str,
    nist_mz,
    nist_int,
):
    """
    Adds ML correction probability to each assignment object.
    Modifies assignments in-place.
    """
    for a in assignments:
        if a.best_formula is None or a.best_exact_mz is None:
            a.ml_prob = 0.0
            continue

        p = ML_MODEL.predict_prob(
            parent_name=parent_name,
            parent_formula_str=parent_formula_str,
            nist_mz=nist_mz,
            nist_int=nist_int,
            assignment=a,
        )

        a.ml_prob = float(p)

    return assignments