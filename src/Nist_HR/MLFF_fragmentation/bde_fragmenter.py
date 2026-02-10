# bde_fragmenter.py

import math
from rdkit import Chem
from formula import Formula
from src.Nist_HR.chemistry import exact_mass


# ---------------------------------------------------------
# Utility: Convert RDKit Mol → Formula
# ---------------------------------------------------------

def mol_to_formula(mol):
    """Convert RDKit Mol (explicit H) to Formula object."""
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    return Formula(atom_counts)


# ---------------------------------------------------------
# Utility: Break bonds and return fragments
# ---------------------------------------------------------

def break_bonds(mol, bonds_to_break):
    """Return fragments after breaking specified bonds."""
    rw = Chem.RWMol(mol)
    for bidx in bonds_to_break:
        bond = rw.GetBondWithIdx(bidx)
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True)


# ---------------------------------------------------------
# Utility: Intensity estimation from BDE
# ---------------------------------------------------------

def estimate_intensity(bde, softness=30.0):
    """
    Simple exponential intensity model.
    Lower BDE → higher intensity.
    """
    return math.exp(-bde / softness)


# ---------------------------------------------------------
# One-step fragmentation (break all bonds < threshold)
# ---------------------------------------------------------

def fragment_by_bde(mol, bond_data, threshold=100.0):
    """
    Break all bonds with BDE < threshold.
    Returns fragments with:
      - explicit-H RDKit Mol
      - clean SMILES
      - exact monoisotopic mass
      - formula
      - atom indices
    """
    if mol is None:
        return []

    bonds_to_break = [b["bond_index"] for b in bond_data if b["bde"] < threshold]
    if not bonds_to_break:
        return []

    frags = break_bonds(mol, bonds_to_break)

    results = []
    for f in frags:
        formula = mol_to_formula(f)
        mass = exact_mass(formula, charged=True)
        smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(f), canonical=True)

        results.append({
            "mol": f,
            "smiles": smiles_clean,
            "formula": str(formula),
            "mass": mass,
            "atom_indices": [a.GetIdx() for a in f.GetAtoms()]
        })

    return results


# ---------------------------------------------------------
# Recursive fragmentation tree
# ---------------------------------------------------------

def recursive_fragment(mol, bond_data, depth=0, max_depth=3, threshold=100.0, softness=25.0):
    """
    Recursively fragment a molecule based on BDE.
    Returns a fragmentation tree node:
    {
        "smiles": ...,
        "formula": ...,
        "mass": ...,
        "children": [...]
    }
    """
    # Compute parent node info
    formula = mol_to_formula(mol)
    mass = exact_mass(formula, charged=True)
    smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)

    node = {
        "smiles": smiles_clean,
        "formula": str(formula),
        "mass": mass,
        "children": []
    }

    if depth >= max_depth:
        return node

    # Identify weak bonds
    weak_bonds = [b for b in bond_data if b["bde"] < threshold]
    if not weak_bonds:
        return node

    # Sort by BDE ascending
    weak_bonds = sorted(weak_bonds, key=lambda x: x["bde"])

    children = []

    for b in weak_bonds:
        # Break only this bond
        frags = break_bonds(mol, [b["bond_index"]])

        for f in frags:
            f_formula = mol_to_formula(f)
            f_mass = exact_mass(f_formula, charged=True)
            f_smiles = Chem.MolToSmiles(Chem.RemoveHs(f), canonical=True)

            # EI intensity model (70 eV calibrated)
            intensity = math.exp(-b["bde"] / softness)

            # Recompute bond data for fragment (placeholder BDEs)
            f_bond_data = []
            for bond in f.GetBonds():
                f_bond_data.append({
                    "bond_index": bond.GetIdx(),
                    "atom1": bond.GetBeginAtomIdx(),
                    "atom2": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                    "bde": 999.0  # placeholder until MACE-BDE is run on fragments
                })

            # Recursive step
            subtree = recursive_fragment(
                f, f_bond_data,
                depth=depth + 1,
                max_depth=max_depth,
                threshold=threshold,
                softness=softness
            )

            child_node = {
                "smiles": f_smiles,
                "formula": str(f_formula),
                "mass": f_mass,
                "intensity": intensity,
                "children": subtree["children"]
            }

            children.append(child_node)

    # Normalize intensities
    total = sum(c["intensity"] for c in children)
    if total > 0:
        for c in children:
            c["intensity"] /= total

    node["children"] = children
    return node