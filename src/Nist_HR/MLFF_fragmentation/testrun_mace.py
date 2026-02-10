
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from mace.modules import MACE
from mace.tools import get_default_config
from mace.data import AtomicData

# 1. Load molecule and generate conformer
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.UFFOptimizeMolecule(mol)

# 2. Convert RDKit → MACE AtomicData
def rdkit_to_atomicdata(mol):
    positions = mol.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return AtomicData.from_points(
        pos=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
    )

data = rdkit_to_atomicdata(mol)

# 3. Load pretrained MACE-BDE model
model = MACE.load("mace_bde.model", device="cpu")
model.eval()

# 4. Predict BDEs
with torch.no_grad():
    output = model(data)

bde_values = output["bde"]  # one value per bond