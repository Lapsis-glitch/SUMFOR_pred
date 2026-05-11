# bde_fragmenter.py

import math
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   # silence sanitisation warnings
from formula import Formula
from chemistry import exact_mass


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
    """
    Return fragments after breaking specified bonds.

    Returns
    -------
    frags : list of RDKit Mol
        Fragment molecules.
    frag_atom_maps : list of tuple[int]
        For each fragment, the original atom indices that belong to it.
    """
    rw = Chem.RWMol(mol)
    for bidx in bonds_to_break:
        bond = rw.GetBondWithIdx(bidx)
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    new_mol = rw.GetMol()

    # Get atom index mappings (which original atoms belong to which fragment)
    frag_atom_tuples = Chem.GetMolFrags(new_mol)

    frags = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=False)

    clean_frags = []
    for f in frags:
        try:
            Chem.SanitizeMol(f)
        except Exception:
            # fallback: try sanitizing with flags
            try:
                Chem.SanitizeMol(f, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass  # last resort: leave unsanitized
        clean_frags.append(f)

    return clean_frags, frag_atom_tuples



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

def fragment_by_bde(mol, bond_data, threshold=120.0):
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

    frags, _ = break_bonds(mol, bonds_to_break)

    results = []
    for f in frags:
        formula = mol_to_formula(f)
        mass = exact_mass(formula, charged=True)
        smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(f), canonical=True)

        results.append({
            "mol": f,
            "smiles": smiles_clean,
            "formula": formula,
            "mass": mass,
            "atom_indices": [a.GetIdx() for a in f.GetAtoms()]
        })

    return results


# ---------------------------------------------------------
# Recursive fragmentation tree
# ---------------------------------------------------------

def recursive_fragment(mol, bond_data, depth=0, max_depth=8, threshold=120.0, softness=25.0,
                       _visited=None):
    """
    Recursively fragment a molecule based on BDE.
    Returns a fragmentation tree node:
    {
        "smiles": ...,
        "formula": ...,
        "mass": ...,
        "children": [...]
    }

    A module-level ``_visited`` set (keyed by canonical SMILES) prunes
    duplicate sub-trees that arise when the same fragment is reached
    via different bond-breaking orders.
    """
    if _visited is None:
        _visited = set()

    # Compute parent node info
    formula = mol_to_formula(mol)
    mass = exact_mass(formula, charged=True)
    try:
        smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, kekuleSmiles=False)
    except Exception:
        try:
            smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, kekuleSmiles=False)
        except Exception:
            smiles_clean = "[UNSANITIZED]"

    node = {
        "smiles": smiles_clean,
        "formula": formula,
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
        frags, frag_atom_maps = break_bonds(mol, [b["bond_index"]])

        for f, atom_map in zip(frags, frag_atom_maps):
            # ── Deduplicate: skip if we already expanded this fragment ──
            try:
                f_smiles = Chem.MolToSmiles(f, canonical=True, kekuleSmiles=False)
            except Exception:
                try:
                    f_smiles = Chem.MolToSmiles(Chem.RemoveHs(f), canonical=True, kekuleSmiles=False)
                except Exception:
                    f_smiles = "[UNSANITIZED]"

            if f_smiles in _visited:
                # Still record as a leaf child (no recursion) so the
                # fragment itself isn't lost, just its subtree.
                f_formula = mol_to_formula(f)
                f_mass = exact_mass(f_formula, charged=True)
                intensity = math.exp(-b["bde"] / softness)
                children.append({
                    "smiles": f_smiles,
                    "formula": f_formula,
                    "mass": f_mass,
                    "intensity": intensity,
                    "children": []
                })
                continue

            _visited.add(f_smiles)

            f_formula = mol_to_formula(f)
            f_mass = exact_mass(f_formula, charged=True)

            # EI intensity model (70 eV calibrated)
            intensity = math.exp(-b["bde"] / softness)

            # A6: Propagate parent BDE data for surviving bonds
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_map)}

            f_bond_data = []
            for parent_bond in bond_data:
                a1 = parent_bond["atom1"]
                a2 = parent_bond["atom2"]
                if a1 in old_to_new and a2 in old_to_new:
                    new_a1 = old_to_new[a1]
                    new_a2 = old_to_new[a2]
                    frag_bond = f.GetBondBetweenAtoms(new_a1, new_a2)
                    if frag_bond is not None:
                        f_bond_data.append({
                            "bond_index": frag_bond.GetIdx(),
                            "atom1": new_a1,
                            "atom2": new_a2,
                            "bond_type": parent_bond["bond_type"],
                            "bde": parent_bond["bde"],
                        })

            # Recursive step
            subtree = recursive_fragment(
                f, f_bond_data,
                depth=depth + 1,
                max_depth=max_depth,
                threshold=threshold,
                softness=softness,
                _visited=_visited,
            )

            child_node = {
                "smiles": f_smiles,
                "formula": f_formula,
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