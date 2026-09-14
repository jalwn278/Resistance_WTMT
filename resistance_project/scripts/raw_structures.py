from pathlib import Path
import shutil
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wt-pdb",
        required=True
    )
    parser.add_argument(
        "--mt-pdb",
        required=True,
    )
    parser.add_argument(
        "--candidates-csv",
        required=True,
    )
    parser.add_argument(
        "--raw-root",
        required=True,
    )

    return parser.parse_args()

def store_raw_structures(wt_pdb_path, mt_pdb_path, candidates_csv, raw_root):
    wt_pdb_path = Path(wt_pdb_path).expanduser()
    mt_pdb_path = Path(mt_pdb_path).expanduser()
    candidates_csv = Path(candidates_csv).expanduser()
    raw_root = Path(raw_root).expanduser()

    for pdb_path in (wt_pdb_path,mt_pdb_path):
        if not pdb_path.exists():
            raise FileNotFoundError(
                f"PDB file not found: {pdb_path}"
            )

        if not pdb_path.is_file():
            raise ValueError(
                f"Expected file: {pdb_path}"
            )

        if pdb_path.suffix.lower() != ".pdb":
            raise ValueError(
                f"Expected .pdb: {pdb_path}"
            )
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
            raise ValueError("Candidates CSV is empty")

    wt_target = first_row["target1"]
    mt_target = first_row["target2"]
    pair_dir = raw_root / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    wt_output_path = (pair_dir / f"{wt_target}.pdb")
    mt_output_path = (pair_dir / f"{mt_target}.pdb")
    shutil.copy2(wt_pdb_path, wt_output_path)
    shutil.copy2(mt_pdb_path, mt_output_path)

    return (wt_output_path, mt_output_path)

def main():
    args = parse_args()
    wt_path, mt_path = store_raw_structures(wt_pdb_path=args.wt_pdb, mt_pdb_path=args.mt_pdb, candidates_csv=args.candidates_csv, raw_root=args.raw_root,)

    print(f"WT PDB: {wt_path}")
    print(f"MT PDB: {mt_path}")

if __name__ == "__main__":
    main()