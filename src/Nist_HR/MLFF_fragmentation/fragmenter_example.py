import json
from rdkit import Chem
from bde_fragmenter import recursive_fragment


FAKE_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"


def load_fake_dataset():
    with open(FAKE_JSON, "r") as f:
        return json.load(f)


def mol_from_entry(entry):
    inchi = entry["pubchem"]["inchi"]
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"Failed to parse InChI: {inchi}")
    mol = Chem.AddHs(mol)
    return mol


def bond_data_from_entry(entry):
    """
    Return a list of bond dicts with non-null BDEs,
    exactly in the format recursive_fragment expects.
    """
    bonds = entry["bde_data"]["bonds"]
    # keep only bonds with a numeric BDE
    return [b for b in bonds if b["bde"] is not None]


def run_fragmentation(entry_id, entry):
    mol = mol_from_entry(entry)
    bond_data = bond_data_from_entry(entry)

    tree = recursive_fragment(
        mol,
        bond_data,
        max_depth=2,
        threshold=100.0,
    )

    return tree


if __name__ == "__main__":
    data = load_fake_dataset()

    entry_id = "7"
    entry = data[entry_id]
    print(f"\n=== Fragmentation for entry {entry_id} ===")
    tree = run_fragmentation(entry_id, entry)
    print(tree)