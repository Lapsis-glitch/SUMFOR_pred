# generate_bde_from_pubchem_json_alfabet.py

import json
import time
from rdkit import Chem
from alfabet import model as alfabet_model
import pandas as pd

INPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem.json"
OUTPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"


def get_smiles_from_pubchem(pubchem):
    """
    Extract a SMILES string from PubChem data.
    Priority:
      1) isomeric_smiles
      2) canonical_smiles
      3) inchi → convert to smiles
    """
    if not pubchem:
        return None, None

    smiles = pubchem.get("isomeric_smiles") or pubchem.get("canonical_smiles")
    if smiles:
        return smiles, {"type": "smiles", "value": smiles}

    inchi = pubchem.get("inchi")
    if inchi:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            return smiles, {"type": "inchi", "value": inchi}

    return None, None


def alfabet_predict_single(smiles):
    """
    Call ALFABET on a single SMILES.
    Returns a DataFrame with columns:
      ['molecule', 'bond_index', 'bond_type', 'fragment1', 'fragment2',
       'is_valid_stereo', 'bde_pred', 'bdfe_pred', 'is_valid',
       'bde', 'bdfe', 'set']
    """
    df = alfabet_model.predict([smiles])
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Filter to this molecule (column name is 'molecule')
    df = df[df["molecule"] == smiles].reset_index(drop=True)
    return df


def bde_dict_from_df(df):
    """
    Build {bond_index: bde_pred} from ALFABET output.
    """
    return {
        int(row["bond_index"]): float(row["bde_pred"])
        for _, row in df.iterrows()
        if pd.notnull(row["bde_pred"])
    }


def annotate_bonds_with_bde(mol, bde_dict):
    atoms = [
        {
            "index": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
        }
        for atom in mol.GetAtoms()
    ]

    bonds = []
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        atom1 = mol.GetAtomWithIdx(a1)
        atom2 = mol.GetAtomWithIdx(a2)

        bonds.append({
            "bond_index": idx,
            "atom1": a1,
            "atom2": a2,
            "atom1_symbol": atom1.GetSymbol(),
            "atom2_symbol": atom2.GetSymbol(),
            "bond_type": str(bond.GetBondType()),
            "bde": bde_dict.get(idx),  # None if ALFABET didn't predict this bond
        })

    return {"atoms": atoms, "bonds": bonds}


def main():
    print("Loading PubChem-augmented dataset...")
    with open(INPUT_JSON, "r") as f:
        merged = json.load(f)

    print("Processing entries with ALFABET (per-molecule)...")
    count = 0
    for entry_id, entry in merged.items():
        count +=1
        if count % 100 == 0:
            print(f"Processing entry {count}/{len(merged)} (ID: {entry_id})")
        pubchem = entry.get("pubchem")
        if not pubchem:
            entry["bde_data"] = None
            continue

        smiles, structure_source = get_smiles_from_pubchem(pubchem)
        if not smiles:
            entry["bde_data"] = None
            continue

        try:
            df_single = alfabet_predict_single(smiles)
            per_bond_bde = bde_dict_from_df(df_single)
        except Exception as e:
            print(f"ALFABET failed for entry {entry_id}: {e}")
            entry["bde_data"] = None
            continue

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        bde_struct = annotate_bonds_with_bde(mol, per_bond_bde)

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