import copy
import sys
from pathlib import Path
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.torsion import get_torsion_angles

smiles = "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC"

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    return mol

def generate_3d(mol):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol,params)

    if status != 0:
        raise ValueError("Failed to generate 3D conformer")
    AllChem.UFFOptimizeMolecule(mol)

    return mol

def calculate_geo(mol, torsions):
    conf = mol.GetConformer()
    geo = []
    for torsion in torsions:
        angle = rdMolTransforms.GetDihedralRad(conf,*torsion)
        geo.append(round(angle, 2))

    return geo

mol = smiles_to_mol(smiles)
mol = generate_3d(mol)
print("Conformer number:",mol.GetNumConformers())
torsions = get_torsion_angles(mol)
geo = calculate_geo(mol,torsions)

print("Number of torsions:", len(torsions))
print("GEO:")
print(geo)