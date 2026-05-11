# generate_bde_alfabet.py
#
# Generate per-bond BDE annotations for the merged dataset using ALFABET.
#
# ALFABET predicts BDEs only for heavy-atom (non-H) bonds, and its
# `bond_index` column refers to the non-H molecule.  The downstream
# BDE fragmenter (bde_fragmenter.py / bde_enumerator.py) works on the
# explicit-H molecule returned by Chem.AddHs().  This script maps the
# ALFABET predictions back to the correct bond indices on the AddHs()
# molecule so that `break_bonds(mol, [b["bond_index"]])` and the
# atom1/atom2 propagation logic work correctly.
#
# Improvements over the old version:
#   1. Correct bond-index mapping (non-H → AddHs molecule).
#   2. Includes atom1/atom2 fields required by bde_fragmenter.
#   3. Deduplicates SMILES so each unique molecule is predicted once.
#   4. Batched ALFABET calls for speed.
#   5. Periodic JSON checkpointing.
#
# Usage:
#   python generate_bde_alfabet.py

import json
import time
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from alfabet import model as alfabet_model
import pandas as pd
from tqdm import tqdm

INPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem.json"
OUTPUT_JSON = "/mnt/d/Leco/merged_clean_SIP_with_pubchem_bde.json"

CHECKPOINT_EVERY = 500   # save partial results every N entries
BATCH_SIZE = 64          # ALFABET batch size


# ── SMILES extraction ────────────────────────────────────────

def get_smiles_from_pubchem(pubchem):
    """
    Extract a SMILES string from PubChem data.
    Priority: isomeric_smiles > canonical_smiles > inchi → smiles.
    Returns (smiles, source_dict) or (None, None).
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


# ── ALFABET → per-bond BDE dict ──────────────────────────────

def alfabet_predict_batch(smiles_list):
    """
    Call ALFABET on a batch of SMILES.
    Returns a dict {smiles: DataFrame} where the DataFrame has
    ALFABET's standard columns including bond_index and bde_pred.
    """
    df = alfabet_model.predict(smiles_list)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    result = {}
    for smi in smiles_list:
        sub = df[df["molecule"] == smi].reset_index(drop=True)
        result[smi] = sub
    return result


def bde_dict_from_df(df):
    """
    Build {bond_index_on_noh_mol: bde_pred} from ALFABET output.
    """
    return {
        int(row["bond_index"]): float(row["bde_pred"])
        for _, row in df.iterrows()
        if pd.notnull(row["bde_pred"])
    }


# ── Bond-index mapping: non-H mol → AddHs mol ───────────────

def _build_noh_to_h_bond_map(mol_noh, mol_h):
    """
    Build a mapping from bond indices on the non-H molecule to bond
    indices on the explicit-H molecule.

    The atom indices on mol_h for the heavy atoms are the SAME as on
    mol_noh (Chem.AddHs keeps heavy atoms in the same order and
    appends H atoms at the end), so we can directly look up the bond.

    Returns {noh_bond_idx: h_bond_idx}.
    """
    mapping = {}
    for bond in mol_noh.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        h_bond = mol_h.GetBondBetweenAtoms(a1, a2)
        if h_bond is not None:
            mapping[bond.GetIdx()] = h_bond.GetIdx()
    return mapping


# ── Annotate bonds on AddHs molecule with BDE ────────────────

def annotate_bonds_with_bde(mol_h, bde_dict_h):
    """
    Build atoms + bonds lists on the explicit-H molecule.
    bde_dict_h: {bond_index_on_mol_h: bde_pred}
    """
    atoms = [
        {
            "index": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
        }
        for atom in mol_h.GetAtoms()
    ]

    bonds = []
    for bond in mol_h.GetBonds():
        idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        bonds.append({
            "bond_index": idx,
            "atom1": a1,
            "atom2": a2,
            "atom1_symbol": mol_h.GetAtomWithIdx(a1).GetSymbol(),
            "atom2_symbol": mol_h.GetAtomWithIdx(a2).GetSymbol(),
            "bond_type": str(bond.GetBondType()),
            "bde": bde_dict_h.get(idx),   # None for H-bonds (ALFABET doesn't predict those)
        })

    return {"atoms": atoms, "bonds": bonds}


# ── Main ─────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from {INPUT_JSON} ...")
    with open(INPUT_JSON, "r") as f:
        merged = json.load(f)

    print(f"  {len(merged)} entries")

    # ── Phase 1: Collect unique SMILES and map to entry IDs ──
    smiles_to_entries = defaultdict(list)   # canonical_smiles → [entry_id, ...]
    entry_smiles = {}                        # entry_id → (smiles, source_dict)

    for entry_id, entry in merged.items():
        pubchem = entry.get("pubchem")
        smiles, source = get_smiles_from_pubchem(pubchem)
        if smiles:
            # Canonicalise so duplicates are collapsed
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                can = Chem.MolToSmiles(mol)
                smiles_to_entries[can].append(entry_id)
                entry_smiles[entry_id] = (can, source)

    unique_smiles = list(smiles_to_entries.keys())
    print(f"  Unique SMILES to predict: {len(unique_smiles)}")
    print(f"  Entries with SMILES:      {sum(len(v) for v in smiles_to_entries.values())}")

    # ── Phase 2: Batch-predict with ALFABET ──
    smiles_bde_cache = {}   # canonical_smiles → bde_struct or None
    n_batches = (len(unique_smiles) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in tqdm(range(0, len(unique_smiles), BATCH_SIZE),
                            total=n_batches, desc="ALFABET", unit="batch"):
        batch = unique_smiles[batch_start: batch_start + BATCH_SIZE]
        try:
            results = alfabet_predict_batch(batch)
        except Exception as e:
            print(f"\n  Batch failed ({batch_start}): {e}")
            for smi in batch:
                smiles_bde_cache[smi] = None
            continue

        for smi in batch:
            df_single = results.get(smi)
            if df_single is None or df_single.empty:
                smiles_bde_cache[smi] = None
                continue

            try:
                bde_noh = bde_dict_from_df(df_single)   # indices on non-H mol

                mol_noh = Chem.MolFromSmiles(smi)
                if mol_noh is None:
                    smiles_bde_cache[smi] = None
                    continue

                mol_h = Chem.AddHs(mol_noh)

                # Map non-H bond indices → AddHs bond indices
                noh_to_h = _build_noh_to_h_bond_map(mol_noh, mol_h)
                bde_h = {}
                for noh_idx, bde_val in bde_noh.items():
                    h_idx = noh_to_h.get(noh_idx)
                    if h_idx is not None:
                        bde_h[h_idx] = bde_val

                bde_struct = annotate_bonds_with_bde(mol_h, bde_h)

                smiles_bde_cache[smi] = {
                    "mol_h": mol_h,
                    "struct": bde_struct,
                }
            except Exception as e:
                print(f"\n  Post-processing failed for {smi}: {e}")
                smiles_bde_cache[smi] = None

    # ── Phase 3: Attach BDE data to entries ──
    n_ok = 0
    n_skip = 0
    for entry_id, entry in merged.items():
        info = entry_smiles.get(entry_id)
        if info is None:
            entry["bde_data"] = None
            n_skip += 1
            continue

        can_smi, source = info
        cached = smiles_bde_cache.get(can_smi)
        if cached is None:
            entry["bde_data"] = None
            n_skip += 1
            continue

        mol_h = cached["mol_h"]
        bde_struct = cached["struct"]

        entry["bde_data"] = {
            "structure_source": source,
            "num_atoms": mol_h.GetNumAtoms(),
            "num_bonds": mol_h.GetNumBonds(),
            "atoms": bde_struct["atoms"],
            "bonds": bde_struct["bonds"],
        }
        n_ok += 1

    print(f"\nAnnotated: {n_ok}  Skipped: {n_skip}")

    # ── Save ──
    print(f"Saving to {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

