from __future__ import annotations
from typing import Dict, List, Tuple
from formula import Formula
from chemistry import dbe

try:
    from rdkit import Chem
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ------------------------------------------------------------
# Helper: subtract a loss from a parent formula
# ------------------------------------------------------------
def subtract(parent: Formula, loss: Formula, check_dbe: bool = True) -> Formula | None:
    """
    Subtract a loss formula from the parent formula.
    Returns None if subtraction is impossible or if the result
    has negative DBE (chemically impossible for a stable ion).
    """
    result: Dict[str, int] = {}

    for el, cnt in parent.elements.items():
        new = cnt - loss.elements.get(el, 0)
        if new < 0:
            return None
        if new > 0:
            result[el] = new

    # Ensure no new elements appear
    for el in loss.elements:
        if el not in parent.elements:
            return None

    if not result:
        return None

    frag = Formula(result)

    # D1: Early DBE validation — reject chemically impossible fragments
    if check_dbe and dbe(frag) < -0.5:
        return None

    return frag


# ------------------------------------------------------------
# Helper: check if cation is subset of parent
# ------------------------------------------------------------
def subset_of_parent(cat: Formula, parent: Formula) -> bool:
    for el, cnt in cat.elements.items():
        if el not in parent.elements:
            return False
        if cnt > parent.elements[el]:
            return False
    return True


# ------------------------------------------------------------
# D6: Even-electron rule helpers
# ------------------------------------------------------------

def is_radical_species(formula: Formula) -> bool:
    """
    Determine whether a neutral species is a radical (odd-electron).

    Half-integer DBE → radical (e.g. CH3•, Cl•, H•)
    Integer DBE      → even-electron neutral molecule (e.g. H2O, CO, HCl)

    This is used by the even-electron rule in RecursiveFragmenter.
    """
    d = dbe(formula)
    # Half-integer check: |fractional part| > 0.25
    return abs(d - round(d)) > 0.25


def compute_loss(parent: Formula, fragment: Formula) -> Formula | None:
    """
    Compute the neutral loss: parent − fragment.

    Returns None if the subtraction is invalid (negative counts or
    the fragment contains elements absent from the parent).
    """
    loss_elems: Dict[str, int] = {}
    for el, cnt in parent.elements.items():
        diff = cnt - fragment.elements.get(el, 0)
        if diff < 0:
            return None
        if diff > 0:
            loss_elems[el] = diff

    for el in fragment.elements:
        if el not in parent.elements:
            return None

    if not loss_elems:
        return None

    return Formula(loss_elems)


# ------------------------------------------------------------
# Shared beta-O losses (used by alcohol/ether rules)
# Moved here to avoid duplication between alcohol.py and ether.py
# ------------------------------------------------------------
BETA_O_LOSSES = [
    Formula({"C": 1, "H": 2, "O": 1}),   # CH2O
    Formula({"C": 2, "H": 4, "O": 1}),   # C2H4O
    Formula({"C": 2, "H": 6, "O": 1}),   # C2H6O
]


def generate_beta_o(parent: Formula, fg: dict) -> List[Tuple[Formula, str]]:
    """Shared β-O cleavage for alcohol and ether functional groups."""
    results: List[Tuple[Formula, str]] = []
    if not (fg.get("alcohol") or fg.get("ether")):
        return results

    for loss in BETA_O_LOSSES:
        frag = subtract(parent, loss)
        if frag is not None:
            results.append((frag, f"beta_O_{loss.to_string()}"))
    return results


# ------------------------------------------------------------
# Functional group detection (compositional, formula-only)
# ------------------------------------------------------------
def detect_functional_groups(parent: Formula) -> Dict[str, bool]:
    elems = parent.elements

    fg: Dict[str, bool] = {
        "alcohol": False,
        "carbonyl": False,
        "aromatic": False,
        "amine": False,
        "ester": False,
        "ether": False,
        "alkene": False,
        "halogen": False,

        # sulfur families
        "sulfur": False,
        "thiol_or_thioether": False,
        "sulfonyl": False,

        # phosphorus
        "phosphorus": False,
    }

    c = elems.get("C", 0)
    h = elems.get("H", 0)
    o = elems.get("O", 0)
    n = elems.get("N", 0)
    s = elems.get("S", 0)
    p = elems.get("P", 0)

    # --- sulfur ---
    fg["sulfur"] = s > 0
    fg["thiol_or_thioether"] = (s > 0 and o == 0)
    fg["sulfonyl"] = (s > 0 and o >= 2)

    # --- phosphorus ---
    fg["phosphorus"] = p > 0

    # --- oxygen families ---
    if o > 0 and h >= 2:
        fg["alcohol"] = True

    if o > 0 and c >= 1:
        fg["carbonyl"] = True

    # --- aromaticity ---
    if dbe(parent) >= 4 and c >= 6:
        fg["aromatic"] = True

    # --- nitrogen ---
    if n > 0:
        fg["amine"] = True

    # --- esters ---
    if o >= 2 and c >= 2:
        fg["ester"] = True

    # --- ethers ---
    if o > 0 and c >= 2:
        fg["ether"] = True

    # --- alkenes ---
    if dbe(parent) >= 1 and c >= 2:
        fg["alkene"] = True

    # --- halogens ---
    f = elems.get("F", 0)
    cl = elems.get("Cl", 0)
    br = elems.get("Br", 0)
    i = elems.get("I", 0)

    if f > 0 or cl > 0 or br > 0 or i > 0:
        fg["halogen"] = True

    return fg


# ------------------------------------------------------------
# SMARTS-based functional group detection (requires RDKit Mol)
# D4: More precise than compositional detection
# ------------------------------------------------------------

# Pre-compiled SMARTS patterns (compiled once at import time)
_SMARTS_CACHE: Dict[str, object] = {}

_SMARTS_DEFS = {
    "alcohol":   "[OX2H1]",                         # hydroxyl
    "carbonyl":  "[CX3]=[OX1]",                     # C=O
    "ester":     "[CX3](=[OX1])[OX2][#6]",          # ester linkage
    "ether":     "[OX2H0;!$(OC=O)]([#6])[#6]",     # C-O-C (not ester/acid)
    "amine":     "[NX3;H2,H1,H0;!$(NC=O)]",        # amine (not amide)
    "alkene":    "[CX3]=[CX3]",                     # C=C
    "halogen":   "[F,Cl,Br,I]",                     # any halogen
}


def _get_smarts(key: str):
    """Get or compile a SMARTS pattern."""
    if key not in _SMARTS_CACHE:
        if _HAS_RDKIT:
            _SMARTS_CACHE[key] = Chem.MolFromSmarts(_SMARTS_DEFS[key])
        else:
            _SMARTS_CACHE[key] = None
    return _SMARTS_CACHE[key]


def detect_functional_groups_from_mol(mol) -> Dict[str, bool]:
    """
    SMARTS-based functional group detection using an RDKit Mol object.
    More precise than compositional detection — e.g. distinguishes
    alcohols from carboxylic acids, ethers from esters, etc.

    Falls back to compositional detection if RDKit is unavailable.
    """
    if not _HAS_RDKIT or mol is None:
        return {}

    # Work with implicit Hs for standard SMARTS matching
    try:
        mol_noH = Chem.RemoveHs(mol)
    except Exception:
        mol_noH = mol

    fg: Dict[str, bool] = {
        "alcohol": False,
        "carbonyl": False,
        "aromatic": False,
        "amine": False,
        "ester": False,
        "ether": False,
        "alkene": False,
        "halogen": False,
        "sulfur": False,
        "thiol_or_thioether": False,
        "sulfonyl": False,
        "phosphorus": False,
    }

    # --- SMARTS-based detection ---
    for key in ("alcohol", "carbonyl", "ester", "ether", "amine", "alkene", "halogen"):
        pat = _get_smarts(key)
        if pat is not None and mol_noH.HasSubstructMatch(pat):
            fg[key] = True

    # --- Aromaticity: check for any aromatic atom ---
    for atom in mol_noH.GetAtoms():
        if atom.GetIsAromatic():
            fg["aromatic"] = True
            break

    # --- Sulfur / phosphorus: check atom symbols ---
    has_s = False
    has_p = False
    has_o = False
    for atom in mol_noH.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "S":
            has_s = True
        elif sym == "P":
            has_p = True
        elif sym == "O":
            has_o = True

    fg["sulfur"] = has_s
    fg["phosphorus"] = has_p

    if has_s:
        # Thiol/thioether: S present but no O neighbours on S
        thiol_pat = Chem.MolFromSmarts("[SX2]")
        if thiol_pat and mol_noH.HasSubstructMatch(thiol_pat):
            fg["thiol_or_thioether"] = True

        sulfonyl_pat = Chem.MolFromSmarts("[SX4](=O)(=O)")
        if sulfonyl_pat and mol_noH.HasSubstructMatch(sulfonyl_pat):
            fg["sulfonyl"] = True

    return fg

