import csv
import copy
import argparse
from pathlib import Path
from rdkit.Chem import AllChem,rdMolTransforms
import numpy as np
import networkx as nx
from rdkit import Chem

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", 
                        required=True, 
                        help="Path to generated molecule CSV")
    parser.add_argument("--output-dir",
                        required=True,
                        help="Directory for reconstructed SDF files")
    parser.add_argument("--force",
                        action="store_true",
                        help="Overwrite existing SDF files")

    return parser.parse_args()

def read_csv(csv_path):
    csv_path = Path(csv_path).expanduser()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV does not exist: {csv_path}"
        )
    if not csv_path.is_file():
        raise IsADirectoryError(
            f"CSV path is not a file: {csv_path}"
        )
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected .csv file: {csv_path}"
        )

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Smiles", "GEO", "target1", "target2"}
        columns = set(reader.fieldnames or [])
        missing_columns = required_columns - columns

        if missing_columns:
            raise ValueError(
                f"Missing columns {sorted(missing_columns)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no molecule rows")

    return rows

def parse_geo(geo_text):
    geo_text = geo_text.strip()
    if not geo_text:
        raise ValueError("GEO is empty")

    try:
        angles = np.array(geo_text.split(), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Invalid GEO values: {geo_text}") from exc

    return angles

def parse_smiles(smiles):
    smiles = smiles.strip()
    if not smiles:
        raise ValueError("SMILES is empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol

def build_molecule_graph(mol):
    graph = nx.Graph()
    for index, atom in enumerate(mol.GetAtoms()):
        graph.add_node(index)
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        graph.add_edge(start,end)

    return graph

def fragment_large_checking(graph_copy):
    components = sorted(nx.connected_components(graph_copy), key=len)
    small_fragment = components[0]
    if len(small_fragment) < 2:
        return False

    return True

def build_torsion(graph_copy, edge):
    left_neighbors= list(graph_copy.neighbors(edge[0]))
    right_neighbors= list(graph_copy.neighbors(edge[1]))
    torsion = (left_neighbors[0], edge[0], edge[1], right_neighbors[0])

    return torsion

def get_torsion_angles(mol):
    torsions = []
    graph = build_molecule_graph(mol)
    for edge in graph.edges():
        graph_copy = copy.deepcopy(graph)
        graph_copy.remove_edge(*edge)
        if nx.is_connected(graph_copy):
            continue
        if not fragment_large_checking(graph_copy):
            continue

        torsion = build_torsion(graph_copy,edge)
        torsions.append(torsion)

    return torsions

def geo_torsion_match(geo_angles, torsions):
    if len(geo_angles) != len(torsions):
        raise ValueError(f"{len(geo_angles)} GEO angles "
                         f"{len(torsions)} torsions")

    return True

def embed_initial_conformer(mol):
    conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1)
    if len(conformer_ids) == 0:
        raise ValueError('Failed to generate 3D conformer')

    return mol

def set_dihedral(conf, torsion, angle):
    rdMolTransforms.SetDihedralRad(
        conf, torsion[0], torsion[1], torsion[2], torsion[3], angle
    )

def apply_torsion_angles(mol,torsions,geo_angles):
    conf = mol.GetConformer(0)
    for index in range(len(torsions)):
        set_dihedral(conf, torsions[index], geo_angles[index])

    return mol

def get_dihedral(conf, torsion):
    angle = rdMolTransforms.GetDihedralRad(conf, torsion[0], torsion[1], torsion[2], torsion[3])

    return angle

def verify_torsion_angles(mol, torsions, geo_angles):
    conf = mol.GetConformer(0)
    for index in range(len(torsions)):
        a_angle = get_dihedral(conf, torsions[index])
        e_angle = geo_angles[index]
        angle_difference = np.arctan2(
            np.sin(a_angle - e_angle),
            np.cos(a_angle - e_angle)
        )
        if abs(angle_difference) > 1e-6:
            raise ValueError(
                f"Torsion {index} doesn't match GEO: "
                f"expected={e_angle:.6f}, "
                f"actual={a_angle:.6f}, "
                f"difference={angle_difference:.6f}"
            )

    return True

def get_atom_coordinates(mol):
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")
    conf = mol.GetConformer(0)
    coordinates = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        position = conf.GetAtomPosition(index)
        coordinates.append((index, position.x, position.y, position.z))

    return coordinates

def validate_coordinates(mol):
    coordinates = get_atom_coordinates(mol)
    for index, x, y, z in coordinates:
        position = np.array([x, y, z], dtype=np.float64)
        if not np.isfinite(position).all():
            raise ValueError(f"Atom {index} has invalid 3D coordinates")

    return True

def save_sdf(mol, output_path, force=False):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"SDF already exists: {output_path}. "
            f"Use --force to overwrite."
        )

    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()

def output_directory(csv_path, output_root):
    csv_path = Path(csv_path).expanduser()
    output_root = Path(output_root).expanduser()
    generation_id = csv_path.parent.name

    output_dir = output_root / generation_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

def reconstruct_molecule(row):
    mol = parse_smiles(row['Smiles'])
    geo_angles = parse_geo(row["GEO"])

    torsions = get_torsion_angles(mol)
    geo_torsion_match(geo_angles, torsions)

    mol = embed_initial_conformer(mol)
    mol = apply_torsion_angles(mol, torsions, geo_angles)

    verify_torsion_angles(mol, torsions, geo_angles)
    validate_coordinates(mol)

    mol.SetProp("Smiles", row["Smiles"])
    mol.SetProp("GEO", row["GEO"])
    mol.SetProp("target1", row["target1"])
    mol.SetProp("target2", row["target2"])

    info = {
        "n_atoms": mol.GetNumAtoms(),
        "n_geo_angles": len(geo_angles),
        "n_torsions": len(torsions),
    }

    return mol, info

def reconstruct_rows(rows, output_dir, force=False):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for index, row in enumerate(rows, start=1):
        candidate_id = f"molecule_{index:04d}"
        filename = f"{candidate_id}.sdf"
        output_path = output_dir / filename
        try:
            mol, info = reconstruct_molecule(row)
            save_sdf(mol, output_path, force=force)
            report_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "PASS",
                    "Smiles": row["Smiles"],
                    "GEO": row["GEO"],
                    "target1": row["target1"],
                    "target2": row["target2"],
                    "n_atoms": info["n_atoms"],
                    "n_geo_angles": info["n_geo_angles"],
                    "n_torsions": info["n_torsions"],
                    "sdf_path": str(output_path),
                    "error": "",
                }
            )
        except Exception as exc:
            report_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "FAIL",
                    "Smiles": row["Smiles"],
                    "GEO": row["GEO"],
                    "target1": row["target1"],
                    "target2": row["target2"],
                    "n_atoms": "",
                    "n_geo_angles": "",
                    "n_torsions": "",
                    "sdf_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return report_rows

def save_report(report_rows, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'reconstruction_report.csv'
    fieldnames = [
        "candidate_id",
        "status",
        "Smiles",
        "GEO",
        "target1",
        "target2",
        "n_atoms",
        "n_geo_angles",
        "n_torsions",
        "sdf_path",
        "error",
    ]

    with report_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    return report_path        

def main():
    args = parse_args()
    rows = read_csv(args.csv)
    output_dir = output_directory(args.csv,args.output_dir)
    report_rows = reconstruct_rows(rows, output_dir, force=args.force)
    report_path = save_report(report_rows, output_dir)
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()