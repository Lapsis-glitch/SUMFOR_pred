# generate_bde_from_pubchem_json.py

import json
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
# before importing mace
import functools

if not hasattr(functools, "cached_property"):
    class cached_property(property):
        pass
    functools.cached_property = cached_property

if not hasattr(functools, "cache"):
    def cache(func):
        return functools.lru_cache(maxsize=None)(func)
    functools.cache = cache
from mace.data import AtomicData
from mace.modules import MACE
import torch


INPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem.json"
OUTPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"
MACE_MODEL_PATH = "/mnt/d/Leco/MACE-BDE/mace.model"


# ---------------- RDKit 3D conformer generation ----------------

def mol_from_pubchem(pubchem):
    """
    Build RDKit Mol with 3D coordinates from PubChem block.

    Priority:
      1) isomeric_smiles
      2) canonical_smiles
      3) inchi

    Returns (mol, structure_source) or (None, None), where
    structure_source = {"type": "smiles"/"inchi", "value": "..."}.
    """
    if not pubchem:
        return None, None

    smiles = pubchem.get("isomeric_smiles") or pubchem.get("canonical_smiles")
    inchi = pubchem.get("inchi")

    mol = None
    source = None

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        source = {"type": "smiles", "value": smiles}
    elif inchi:
        mol = Chem.MolFromInchi(inchi)
        source = {"type": "inchi", "value": inchi}

    if mol is None:
        return None, None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D

    try:
        AllChem.EmbedMolecule(mol, params)
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None, None

    return mol, source


# ---------------- RDKit Mol → MACE AtomicData ----------------
Z_MAP = {1: 0, 6: 1, 8: 2}

Z_MAP = {1: 0, 6: 1, 8: 2}   # H, C, O → 0,1,2

def rdkit_to_atomicdata(mol, cutoff=5.0):
    # Always use double precision (model parameters are float64)
    dtype = torch.double

    # --- Positions and atomic numbers ---
    conf = mol.GetConformer()
    positions = torch.tensor(conf.GetPositions(), dtype=dtype)   # (n,3)
    atomic_numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    # Reject unsupported atoms
    for z in atomic_numbers:
        if z.item() not in Z_MAP:
            raise ValueError(f"Unsupported element Z={z.item()} for this MACE-BDE model (only H,C,O allowed).")

    n = positions.shape[0]

    # --- Correct one-hot node attributes (n,3) ---
    indices = torch.tensor([Z_MAP[z.item()] for z in atomic_numbers], dtype=torch.long)
    node_attrs = torch.nn.functional.one_hot(indices, num_classes=3).to(dtype)

    # --- Build neighbor list ---
    rij = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.norm(rij, dim=-1)
    mask = (dist < cutoff) & (dist > 1e-6)
    src, dst = torch.where(mask)

    if src.numel() == 0:
        # Ensure at least one edge
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([0], dtype=torch.long)
        edge_vec = torch.zeros((1, 3), dtype=dtype)
    else:
        edge_vec = rij[src, dst]

    edge_index = torch.stack([src, dst], dim=0)

    # --- Shifts and unit shifts (no PBC) ---
    shifts = torch.zeros_like(edge_vec)
    unit_shifts = torch.zeros_like(edge_vec)

    # --- Cell and PBC ---
    cell = torch.zeros((3, 3), dtype=dtype)
    pbc = torch.tensor([[False, False, False]], dtype=torch.bool)

    # --- Scalars required by asserts (must be 0‑dim tensors) ---
    zero = torch.tensor(0.0, dtype=dtype)

    weight = zero
    head = zero
    energy_weight = zero
    forces_weight = zero
    stress_weight = zero
    virials_weight = zero
    charges_weight = zero
    elec_temp = zero
    total_charge = zero
    total_spin = zero

    # --- Vector/tensor weights with required shapes ---
    dipole_weight = torch.zeros((1, 3), dtype=dtype)
    polarizability_weight = torch.zeros((1, 3, 3), dtype=dtype)

    # --- Targets (zeros, correct shapes) ---
    forces = torch.zeros((n, 3), dtype=dtype)
    energy = zero
    stress = torch.zeros((1, 3, 3), dtype=dtype)
    virials = torch.zeros((1, 3, 3), dtype=dtype)
    dipole = torch.zeros((1, 3), dtype=dtype)
    charges = torch.zeros((n,), dtype=dtype)
    polarizability = torch.zeros((1, 3, 3), dtype=dtype)

    # --- Build AtomicData (all asserts satisfied) ---
    data = AtomicData(
        edge_index=edge_index,
        node_attrs=node_attrs,
        positions=positions,
        shifts=shifts,
        unit_shifts=unit_shifts,
        cell=cell,
        weight=weight,
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        dipole_weight=dipole_weight,
        charges_weight=charges_weight,
        polarizability_weight=polarizability_weight,
        forces=forces,
        energy=energy,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        polarizability=polarizability,
        elec_temp=elec_temp,
        total_charge=total_charge,
        total_spin=total_spin,
        pbc=pbc,
    )

    # --- Add batch + ptr (required by prepare_graph) ---
    data.batch = torch.zeros(n, dtype=torch.long)
    data.ptr = torch.tensor([0, n], dtype=torch.long)

    # --- Convert scalars to shape (1,) for prepare_graph ---
    for key in [
        "weight", "head", "energy_weight", "forces_weight", "stress_weight",
        "virials_weight", "charges_weight", "elec_temp", "total_charge", "total_spin"
    ]:
        setattr(data, key, getattr(data, key).unsqueeze(0))

    return data


# ---------------- MACE-BDE prediction ----------------



def load_mace_model():
    """
    Load the MACE-BDE model exactly as saved in the paper's repository.
    The checkpoint is a full model object, not a dict.
    """
    device = torch.device("cuda")
    model = torch.load(MACE_MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()

    return model




def predict_bde(mol, model):
    data = rdkit_to_atomicdata(mol)
    with torch.no_grad():
        out = model(data)
    return out["bde"].tolist()


# ---------------- Bond + atom annotation ----------------

def annotate_bonds_with_bde(mol, bde_values):
    """
    Return a dict with:
      - atoms: list of {index, symbol, atomic_number}
      - bonds: list of {bond_index, atom1, atom2, atom1_symbol, atom2_symbol, bond_type, bde}
    """
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append({
            "index": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
        })

    bonds = []
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        atom1 = mol.GetAtomWithIdx(a1)
        atom2 = mol.GetAtomWithIdx(a2)
        btype = str(bond.GetBondType())

        bonds.append({
            "bond_index": idx,
            "atom1": a1,
            "atom2": a2,
            "atom1_symbol": atom1.GetSymbol(),
            "atom2_symbol": atom2.GetSymbol(),
            "bond_type": btype,
            "bde": bde_values[idx],
        })

    return {
        "atoms": atoms,
        "bonds": bonds,
    }


# ---------------- Main pipeline ----------------

def main():
    print("Loading PubChem-augmented dataset...")
    with open(INPUT_JSON, "r") as f:
        merged = json.load(f)

    print("Loading MACE-BDE model...")
    mace_model = load_mace_model()

    print("Processing entries...")
    for entry_id, entry in merged.items():

        pubchem = entry.get("pubchem")
        if not pubchem:
            entry["bde_data"] = None
            continue

        # 1. RDKit 3D conformer (from SMILES or InChI)
        mol, structure_source = mol_from_pubchem(pubchem)
        if mol is None:
            print(f"RDKit failed for entry {entry_id}")
            entry["bde_data"] = None
            continue

        data = rdkit_to_atomicdata(mol)
        device = torch.device("cuda")
        data = data.to(device)
        out = mace_model(data)
        print(out.keys())

        # 2. MACE-BDE prediction
        try:
            bde_values = predict_bde(mol, mace_model)
        except Exception as e:
            print(f"MACE-BDE failed for entry {entry_id}: {e}")
            entry["bde_data"] = None
            continue

        # 3. Annotate atoms + bonds with BDE
        bde_struct = annotate_bonds_with_bde(mol, bde_values)

        # 4. Store in structured format
        entry["bde_data"] = {
            "structure_source": structure_source,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "atoms": bde_struct["atoms"],
            "bonds": bde_struct["bonds"],
        }

        time.sleep(0.01)

    print("Saving final dataset...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Done. Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()