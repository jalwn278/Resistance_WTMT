from pathlib import Path
import subprocess
import csv
import argparse

from rdkit import Chem
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates-csv",
        required=True
    )
    parser.add_argument(
            "--molecules-dir",
            required=True
    )
    parser.add_argument(
            "--reference-root",
            required=True
    )
    parser.add_argument(
            "--docking-root",
            required=True
    )
    parser.add_argument(
            "--prepare-receptor-bin",
            required=True
    )
    parser.add_argument(
            "--obabel-bin",
            required=True
    )
    parser.add_argument(
            "--prepare-ligand-bin",
            required=True
    )
    parser.add_argument(
            "--qvina-bin",
            required=True
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild receptor and ligand PDBQT files",
    )

    return parser.parse_args()    

def prepare_protein_receptor(pdb_path, output_dir, prepare_receptor_bin, force=False):
    pdb_path = Path(pdb_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    prepare_receptor_bin = Path(prepare_receptor_bin).expanduser()

    if not pdb_path.exists():
        raise FileNotFoundError(
            f"PDB file not found: {pdb_path}"
        )
    if not pdb_path.is_file():
        raise ValueError(
            f"Expected a file: {pdb_path}"
        )
    if pdb_path.suffix.lower() != ".pdb":
        raise ValueError(
            f"Expected .pdb file: {pdb_path}"
        )
    if not prepare_receptor_bin.exists():
        raise FileNotFoundError(
            "prepare_receptor not found: "
            f"{prepare_receptor_bin}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / f"{pdb_path.stem}.pdbqt")
    if output_path.exists() and not force:
        return output_path

    command = [str(prepare_receptor_bin), "-r", str(pdb_path), "-o", str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError("prepare_receptor failed:\n" 
                           f"{result.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(
            "prepare_receptor finished "
            "but PDBQT was not created."
        )

    return output_path

def prepare_pair_receptor(candidates_csv, reference_root, docking_root, prepare_receptor_bin, force=False):
    candidates_csv = Path(candidates_csv).expanduser()
    reference_root = Path(reference_root).expanduser()
    docking_root = Path(docking_root).expanduser()

    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"Candidates CSV not found: "
            f"{candidates_csv}"
        )
    
    pair_id = candidates_csv.parent.name
    with candidates_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    if first_row is None:
        raise ValueError(
            "Candidates CSV is empty."
        )

    wt_target = first_row["target1"]
    mt_target = first_row["target2"]
    pair_dir = (reference_root / pair_id)
    wt_pdb = (pair_dir / wt_target / "protein.pdb")
    mt_pdb = (pair_dir / mt_target / "protein.pdb")
    wt_output_dir = (docking_root / pair_id / wt_target)
    mt_output_dir = (docking_root / pair_id / mt_target)

    wt_pdbqt = prepare_protein_receptor(pdb_path = wt_pdb, output_dir = wt_output_dir, prepare_receptor_bin = prepare_receptor_bin, force=force)
    mt_pdbqt = prepare_protein_receptor(pdb_path = mt_pdb, output_dir = mt_output_dir, prepare_receptor_bin = prepare_receptor_bin, force=force)

    return (wt_pdbqt, mt_pdbqt)

def get_ligand_centroid(ligand_sdf):
    ligand_sdf = Path(ligand_sdf).expanduser()
    if not ligand_sdf.exists():
        raise FileNotFoundError(
            f"Ligand file not found: {ligand_sdf}"
        )
    if not ligand_sdf.is_file():
        raise ValueError(
            f"Expected a file: {ligand_sdf}"
        )
    if ligand_sdf.suffix.lower() != ".sdf":
        raise ValueError(
            f"Expected .sdf file: {ligand_sdf}"
        )

    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError(
            f"Cannot read ligand: {ligand_sdf}"
        )
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Ligand has no 3D coordinates."
        )

    conf = mol.GetConformer(0)
    coordinates = conf.GetPositions()
    centroid = coordinates.mean(axis=0)

    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))

def prepare_generated_ligand(ligand_sdf, output_dir, obabel_bin, prepare_ligand_bin, force=False):
    ligand_sdf = Path(ligand_sdf).expanduser()
    output_dir = Path(output_dir).expanduser()
    obabel_bin = Path(obabel_bin).expanduser()
    prepare_ligand_bin = Path(prepare_ligand_bin).expanduser()

    if not ligand_sdf.exists():
        raise FileNotFoundError(
            f"Ligand SDF not found: "
            f"{ligand_sdf}"
        )
    if not ligand_sdf.is_file():
        raise ValueError(
            f"Expected a file: "
            f"{ligand_sdf}"
        )
    if ligand_sdf.suffix.lower() != ".sdf":
        raise ValueError(
            f"Expected .sdf file: "
            f"{ligand_sdf}"
        )
    if not obabel_bin.exists():
        raise FileNotFoundError(
            f"Open Babel not found: "
            f"{obabel_bin}"
        )
    if not prepare_ligand_bin.exists():
        raise FileNotFoundError(
            f"prepare_ligand not found: "
            f"{prepare_ligand_bin}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ligand_name = ligand_sdf.stem
    mol2_path = (output_dir / f"{ligand_name}.mol2")
    pdbqt_path = (output_dir / f"{ligand_name}.pdbqt")
    if pdbqt_path.exists() and not force:
        return pdbqt_path

    obabel_command = [str(obabel_bin),
                      str(ligand_sdf),
                      "-O",
                      str(mol2_path)]
    obabel_result = subprocess.run(obabel_command,
                                   capture_output=True,
                                   text=True)
    if obabel_result.returncode != 0:
        raise RuntimeError(
            "Open Babel failed:\n"
            f"{obabel_result.stderr}"
        )
    if not mol2_path.exists():
        raise RuntimeError(
            "Open Babel finished "
            "but MOL2 was not created."
        )

    prepare_command = [str(prepare_ligand_bin),
                       "-l",
                       mol2_path.name,
                       "-A",
                       "hydrogens",
                       "-o",
                       pdbqt_path.name]
    prepare_result = subprocess.run(prepare_command,
                                    capture_output=True,
                                    text=True,
                                    cwd=str(output_dir))
    if prepare_result.returncode != 0:
        raise RuntimeError(
            "prepare_ligand failed:\n"
            f"{prepare_result.stderr}"
        )
    if not pdbqt_path.exists():
        raise RuntimeError(
            "prepare_ligand finished "
            "but PDBQT was not created."
        )

    return pdbqt_path

def prepare_generated_ligands(molecules_dir, output_dir, obabel_bin, prepare_ligand_bin, force=False):
    molecules_dir = Path(molecules_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    if not molecules_dir.exists():
        raise FileNotFoundError(
            f"Molecules directory not found: "
            f"{molecules_dir}"
        )
    if not molecules_dir.is_dir():
        raise ValueError(
            f"Expected a directory: "
            f"{molecules_dir}"
        )

    sdf_files = sorted(molecules_dir.glob("*.sdf"))
    if not sdf_files:
        raise ValueError(
            f"No SDF files found in: "
            f"{molecules_dir}"
        )

    pdbqt_files = []
    for ligand_sdf in sdf_files:
        pdbqt_path = prepare_generated_ligand(ligand_sdf=ligand_sdf, output_dir=output_dir, obabel_bin=obabel_bin, prepare_ligand_bin=prepare_ligand_bin, force=force)
        pdbqt_files.append(pdbqt_path)

    return pdbqt_files

def dock_one_ligand(receptor_pdbqt, ligand_pdbqt, centroid, output_dir, qvina_bin, obabel_bin):
    receptor_pdbqt = Path(receptor_pdbqt).expanduser()
    ligand_pdbqt = Path(ligand_pdbqt).expanduser()
    output_dir = Path(output_dir).expanduser()
    qvina_bin = Path(qvina_bin).expanduser()
    obabel_bin = Path(obabel_bin).expanduser()

    if not receptor_pdbqt.exists():
        raise FileNotFoundError(
            f"Receptor PDBQT not found: "
            f"{receptor_pdbqt}"
        )
    if not ligand_pdbqt.exists():
        raise FileNotFoundError(
            f"Ligand PDBQT not found: "
            f"{ligand_pdbqt}"
        )
    if not qvina_bin.exists():
        raise FileNotFoundError(
            f"QVina2 not found: "
            f"{qvina_bin}"
        )
    if not obabel_bin.exists():
        raise FileNotFoundError(
            f"Open Babel not found: "
            f"{obabel_bin}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = centroid
    out_pdbqt = (output_dir / f"{ligand_pdbqt.stem}_out.pdbqt")
    out_sdf = (output_dir / f"{ligand_pdbqt.stem}_out.sdf")

    qvina_command = [str(qvina_bin),
                     "--receptor",
                     str(receptor_pdbqt),
                     "--ligand",
                     str(ligand_pdbqt),
                     "--center_x",
                     str(cx),
                     "--center_y",
                     str(cy),
                     "--center_z",
                     str(cz),
                     "--size_x",
                     "20",
                     "--size_y",
                     "20",
                     "--size_z",
                     "20",
                     "--cpu",
                     "40",
                     "--exhaustiveness",
                     "8",
                     "--out",
                     str(out_pdbqt)]
    qvina_result = subprocess.run(qvina_command, capture_output=True, text=True)
    if qvina_result.returncode != 0:
        raise RuntimeError(
            "QVina2 failed:\n"
            f"{qvina_result.stderr}"
        )
    if not out_pdbqt.exists():
        raise RuntimeError(
            "QVina2 finished "
            "but output PDBQT was not created."
        )

    obabel_command = [str(obabel_bin),
                      str(out_pdbqt),
                      "-O",
                      str(out_sdf),
                      "-h"]
    obabel_result = subprocess.run(obabel_command,
                                   capture_output=True,
                                   text=True)
    if obabel_result.returncode != 0:
        raise RuntimeError(
            "Open Babel failed:\n"
            f"{obabel_result.stderr}"
        )

    if not out_sdf.exists():
        raise RuntimeError(
            "Open Babel finished "
            "but docked SDF was not created."
        )

    return out_sdf

def get_docking_result(docked_sdf):
    docked_sdf = Path(docked_sdf).expanduser()

    if not docked_sdf.exists():
        raise FileNotFoundError(
            f"Docked SDF not found: "
            f"{docked_sdf}"
        )
    if not docked_sdf.is_file():
        raise ValueError(
            f"Expected a file: "
            f"{docked_sdf}"
        )
    if docked_sdf.suffix.lower() != ".sdf":
        raise ValueError(
            f"Expected .sdf file: "
            f"{docked_sdf}"
        )

    supplier = Chem.SDMolSupplier(str(docked_sdf), sanitize=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError(
            f"Cannot read docked molecule: "
            f"{docked_sdf}"
        )
    if not mol.HasProp("REMARK"):
        raise ValueError(
            "Docking result has no "
            "REMARK property"
        )

    remark = mol.GetProp("REMARK")
    first_line = (remark.splitlines()[0])
    fields = first_line.split()
    if len(fields) < 3:
        raise ValueError(
            f"Unexpected docking REMARK:"
            F"{first_line}"
        )
    affinity = float(fields[2])

    return affinity

def dock_ligand_pair(ligand_pdbqt, wt_receptor_pdbqt, mt_receptor_pdbqt, wt_centroid, mt_centroid, wt_output_dir, mt_output_dir, qvina_bin, obabel_bin):
    wt_docked_sdf = dock_one_ligand(receptor_pdbqt=wt_receptor_pdbqt,
                                    ligand_pdbqt=ligand_pdbqt,
                                    centroid=wt_centroid,
                                    output_dir=wt_output_dir,
                                    qvina_bin=qvina_bin,
                                    obabel_bin=obabel_bin)

    mt_docked_sdf = dock_one_ligand(receptor_pdbqt=mt_receptor_pdbqt,
                                    ligand_pdbqt=ligand_pdbqt,
                                    centroid=mt_centroid,
                                    output_dir=mt_output_dir,
                                    qvina_bin=qvina_bin,
                                    obabel_bin=obabel_bin)

    wt_affinity = get_docking_result(wt_docked_sdf)
    mt_affinity = get_docking_result(mt_docked_sdf)

    return (wt_affinity, mt_affinity)

def dock_generated_ligands(candidates_csv, reference_root, docking_root, qvina_bin, obabel_bin):
    candidates_csv = Path(candidates_csv).expanduser()
    reference_root = Path(reference_root).expanduser()
    docking_root = Path(docking_root).expanduser()
    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"Candidates CSV not found: "
            f"{candidates_csv}"
        )

    pair_id = candidates_csv.parent.name
    with candidates_csv.open("r", newline='',encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    if first_row is None:
        raise ValueError(
            "Candidates CSV is empty."
        )
    
    wt_target = first_row["target1"]
    mt_target = first_row["target2"]
    wt_receptor_pdbqt = (docking_root / pair_id / wt_target / "protein.pdbqt")
    mt_receptor_pdbqt = (docking_root / pair_id / mt_target / "protein.pdbqt")
    wt_reference_ligand = (reference_root / pair_id / wt_target / "ligand.sdf")
    mt_reference_ligand = (reference_root / pair_id / mt_target / "ligand.sdf")
    wt_centroid = get_ligand_centroid(wt_reference_ligand)
    mt_centroid = get_ligand_centroid(mt_reference_ligand)
    generated_dir = (docking_root / pair_id / "generated_ligands")
    ligand_files = sorted(generated_dir.glob("*.pdbqt"))
    if not ligand_files:
        raise ValueError(
            f"No generated ligand PDBQT files "
            f"found in: {generated_dir}"
        )

    wt_output_dir = (docking_root / pair_id / wt_target / "docked")
    mt_output_dir = (docking_root / pair_id / mt_target / "docked")
    results = []
    for ligand_pdbqt in ligand_files:
        wt_affinity, mt_affinity = (
            dock_ligand_pair(
                ligand_pdbqt=ligand_pdbqt,
                wt_receptor_pdbqt=(wt_receptor_pdbqt),
                mt_receptor_pdbqt=(mt_receptor_pdbqt),
                wt_centroid=wt_centroid,
                mt_centroid=mt_centroid,
                wt_output_dir=wt_output_dir,
                mt_output_dir=mt_output_dir,
                qvina_bin=qvina_bin,
                obabel_bin=obabel_bin
            )
        )

        results.append(
            {
                "molecule": ligand_pdbqt.stem,
                "wt_target": wt_target,
                "mt_target": mt_target,
                "wt_affinity": wt_affinity,
                "mt_affinity": mt_affinity
            }
        )

    results_csv = (docking_root / pair_id / "docking_results.csv")
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["molecule", "wt_target", "mt_target", "wt_affinity", "mt_affinity"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results_csv

def run_docking(candidates_csv, molecules_dir, reference_root, docking_root, prepare_receptor_bin, obabel_bin, prepare_ligand_bin, qvina_bin, force=False):
    candidates_csv = Path(candidates_csv).expanduser()
    molecules_dir = Path(molecules_dir).expanduser()
    docking_root = Path(docking_root).expanduser()

    pair_id = candidates_csv.parent.name
    generated_dir = (docking_root / pair_id / "generated_ligands")
    prepare_pair_receptor(candidates_csv=candidates_csv,
                          reference_root=reference_root,
                          docking_root=docking_root,
                          prepare_receptor_bin=prepare_receptor_bin,
                          force=force)
    prepare_generated_ligands(molecules_dir=molecules_dir,
                              output_dir=generated_dir,
                              obabel_bin=obabel_bin,
                              prepare_ligand_bin=prepare_ligand_bin,
                              force=force)
    results_csv = dock_generated_ligands(candidates_csv=candidates_csv,
                                         reference_root=reference_root,
                                         docking_root=docking_root,
                                         qvina_bin=qvina_bin,
                                         obabel_bin=obabel_bin)

    return results_csv

def main():
    args = parse_args()
    results_csv = run_docking(candidates_csv=args.candidates_csv,
                              molecules_dir=args.molecules_dir,
                              reference_root=args.reference_root,
                              docking_root=args.docking_root,
                              prepare_receptor_bin=args.prepare_receptor_bin,
                              obabel_bin=args.obabel_bin,
                              prepare_ligand_bin=args.prepare_ligand_bin,
                              qvina_bin=args.qvina_bin,
                              force=args.force)

    print(f"Docking results: {results_csv}")

if __name__ == "__main__":
    main()