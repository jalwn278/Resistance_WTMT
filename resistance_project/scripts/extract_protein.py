from pathlib import Path 
def extract_protein_from_complex(complex_pdb, protein_pdb):
    complex_pdb = Path(complex_pdb).expanduser()
    protein_pdb = Path(protein_pdb).expanduser()

    if not complex_pdb.exists():
        raise FileNotFoundError(
            f"Complex PDB not found: {complex_pdb}"
        )
    protein_pdb.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with complex_pdb.open("r", encoding="utf-8") as source, protein_pdb.open("w", encoding="utf-8") as output:
        for line in source:
            if line.startswith("ATOM"):
                output.write(line)
            elif line.startswith('TER'):
                output.write(line)
        output.write("END\n")

    return protein_pdb

if __name__ == "__main__":

    extract_protein_from_complex(
    "resistance_project/data/docking_reference/8faf60323ce63d5a/P00533_WT/5FED.pdb",
    "resistance_project/data/docking_reference/8faf60323ce63d5a/P00533_WT/protein.pdb",
)