# chemical_classification.py

import re
from formula import Formula
from chemistry import dbe

# ------------------------------------------------------------
# Name-based keyword patterns
# ------------------------------------------------------------

NAME_PATTERNS = {
    r"benz|phenyl|tolu|xyl|naphth": "aromatic",
    r"cyclo": "cyclic",
    r"pyrid|pyrrol|imid|thiaz|oxaz|azole": "heterocycle",

    r"amine|amino": "amine",
    r"amide": "amide",
    r"alcohol|ol$": "alcohol",
    r"ester|acetate|propionate|butyrate": "ester",
    r"ether": "ether",
    r"ketone|one$": "ketone",
    r"aldehyde|al$": "aldehyde",
    r"acid$|carboxy": "acid",
    r"nitrile|cyan": "nitrile",

    r"chloro|bromo|fluoro|iodo": "halogenated",

    r"d[0-9]+$|13c|15n": "isotopically_labeled",
}

# ------------------------------------------------------------
# Formula-based rules
# ------------------------------------------------------------

def classify_from_formula(formula: Formula):
    classes = set()
    elems = formula.elements

    # halogens
    if any(e in elems for e in ["Cl", "Br", "F", "I"]):
        classes.add("halogenated")

    # heteroatoms
    if "N" in elems:
        classes.add("nitrogenous")
    if "O" in elems:
        classes.add("oxygenated")
    if "S" in elems:
        classes.add("sulfur")
    if "P" in elems:
        classes.add("phosphorus")
    if "Si" in elems:
        classes.add("organosilicon")

    # aromaticity hint
    rdbe = dbe(formula)
    c = elems.get("C", 0)
    if c >= 6 and rdbe >= 4:
        classes.add("aromatic")

    return classes

# ------------------------------------------------------------
# Name-based rules
# ------------------------------------------------------------

def classify_from_name(name: str):
    name = name.lower()
    classes = set()

    for pattern, cls in NAME_PATTERNS.items():
        if re.search(pattern, name):
            classes.add(cls)

    return classes

# ------------------------------------------------------------
# Combined classifier
# ------------------------------------------------------------

def classify_molecule(name: str, formula_str: str):
    try:
        formula = Formula.from_string(formula_str)
    except Exception:
        return set()

    classes = set()
    classes |= classify_from_name(name)
    classes |= classify_from_formula(formula)

    # fallback: if nothing detected
    if not classes:
        classes.add("unknown")

    return classes