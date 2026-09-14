import sys
import argparse
import pandas as pd
from pathlib import Path

LAMGEN_ROOT = Path(__file__).resolve().parents[2]

if str(LAMGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(LAMGEN_ROOT))

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from utils.sascore import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import Descriptors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecules-dir",
        required=True,
        help="Directory containing reconstructed SDF molecules"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory for properties CSV files"
    )

    return parser.parse_args()

def read_sdf(sdf_path):
    sdf_path = Path(sdf_path).expanduser()
    if not sdf_path.exists():
        raise FileNotFoundError(
            f"SDF does not exist: {sdf_path}"
        )
    if not sdf_path.is_file():
        raise IsADirectoryError(
            f"SDF path is not a file: {sdf_path}"
        )
    if sdf_path.suffix.lower() != ".sdf":
        raise ValueError(
            f"Expected .sdf file: {sdf_path}"
        )

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    if len(supplier) == 0:
        raise ValueError(f"No molecule found in SDF: {sdf_path}")

    mol = supplier[0]
    if mol is None:
        raise ValueError(f"Failed to parse SDF: {sdf_path}")

    return mol

class Score:
    def __init__(self, mol):
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol)
        self.qed = qed(mol)
        self.sa = sascorer.calculateScore(mol)
        self.logp = MolLogP(mol)
        self.mw = Descriptors.MolWt(mol)
        self.tpsa = CalcTPSA(mol)

def property_record(mol, sdf_path):
    sdf_path = Path(sdf_path)
    score = Score(mol)
    record = {
        "molecule": sdf_path.stem,
        "smiles":score.smiles,
        "target1": mol.GetProp("target1"),
        "target2": mol.GetProp("target2"),
        "QED": score.qed,
        "SA":score.sa,
        "LogP":score.logp,
        "MW":score.mw,
        "TPSA":score.tpsa
    }

    return record

def collect_property_records(molecules_dir):
    molecules_dir = Path(molecules_dir).expanduser()
    if not molecules_dir.exists():
        raise FileNotFoundError(
            f"Molecule directory does not exist: "
            f"{molecules_dir}"
        )
    if not molecules_dir.is_dir():
        raise NotADirectoryError(
            f"Expected directory: {molecules_dir}"
        )

    records = []
    sdf_files = sorted(molecules_dir.glob("*.sdf"))
    if not sdf_files:
        raise ValueError(f"No SDF files found in: {molecules_dir}")

    for sdf_path in sdf_files:
        mol = read_sdf(sdf_path)
        record = property_record(mol, sdf_path)
        records.append(record)

    return records

def save_properties_csv(records, molecules_dir, output_root):
    molecules_dir = Path(molecules_dir).expanduser()
    output_root = Path(output_root).expanduser()
    pair_id = molecules_dir.name
    output_dir = output_root / pair_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        molecule_id = record["molecule"]
        output_path = (output_dir / f"{molecule_id}.csv")
        properties_df = pd.DataFrame([record])
        properties_df.to_csv(output_path, index=False)

def main():
    args = parse_args()
    records = collect_property_records(args.molecules_dir)
    save_properties_csv(records, args.molecules_dir, args.output)

if __name__ == "__main__":
    main()