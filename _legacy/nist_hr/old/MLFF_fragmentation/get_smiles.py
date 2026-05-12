# get_smiles.py

import requests
import json
from time import sleep

def fetch_pubchem_data(inchikey):
    """
    Fetch structural data from PubChem using an InChIKey.
    Returns dict with SMILES, InChI, formula, names, etc.
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
    url = f"{base}/{inchikey}/property/IsomericSMILES,CanonicalSMILES,InChI,MolecularFormula,IUPACName/JSON"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        props = data["PropertyTable"]["Properties"][0]

        return {
            "inchikey": inchikey,
            "canonical_smiles": props.get("CanonicalSMILES"),
            "isomeric_smiles": props.get("IsomericSMILES"),
            "inchi": props.get("InChI"),
            "formula": props.get("MolecularFormula"),
            "iupac_name": props.get("IUPACName"),
        }

    except Exception as e:
        print(f"PubChem lookup failed for {inchikey}: {e}")
        return None



MERGED_CLEAN = "/mnt/d/Leco/merged_clean_SIP.json"
OUT_WITH_PUBCHEM = "/mnt/d/Leco/merged_clean_SIP_with_pubchem.json"

def attach_pubchem_data():
    with open(MERGED_CLEAN, "r") as f:
        merged = json.load(f)

    for entry_id, entry in merged.items():
        inchi_keys = entry.get("inchi_keys", [])
        if not inchi_keys:
            continue

        inchikey = inchi_keys[0]  # best match

        pubchem = fetch_pubchem_data(inchikey)
        entry["pubchem"] = pubchem

        # polite delay to avoid hammering PubChem
        sleep(0.1)

    with open(OUT_WITH_PUBCHEM, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved PubChem-augmented dataset to {OUT_WITH_PUBCHEM}")

if __name__ == "__main__":
    attach_pubchem_data()
