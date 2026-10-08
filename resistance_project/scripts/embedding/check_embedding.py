from pathlib import Path
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a protein embedding stored as a .npy file."
    )

    parser.add_argument(
        "--embedding",
        type=Path,
        required=True,
        help="Path to the .npy embedding file.",
    )

    return parser.parse_args()


def inspect_embedding(path):
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    if not path.is_file():
        raise IsADirectoryError(
            f"Embedding path is not a file: {path.resolve()}"
        )

    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"Embedding file must use the .npy suffix, "
            f"but got: {path.suffix!r}"
        )

    embedding = np.load(
        path,
        allow_pickle=False,
    )

    print("=" * 70)
    print("Embedding inspection")
    print("=" * 70)
    print(f"Path       : {path.resolve()}")
    print(f"Shape      : {embedding.shape}")
    print(f"Dtype      : {embedding.dtype}")
    print(f"Dimensions : {embedding.ndim}")
    print(f"Finite     : {np.isfinite(embedding).all()}")
    print(f"Minimum    : {embedding.min()}")
    print(f"Maximum    : {embedding.max()}")
    print(f"Mean       : {embedding.mean()}")
    print(f"Std        : {embedding.std()}")
    print("Preview    :")
    print(embedding[:3, :10])

    # 检查是否符合 LaMGen 蛋白输入格式
    if embedding.ndim != 2:
        raise ValueError(
            f"Embedding must be 2-dimensional, "
            f"but got shape {embedding.shape}"
        )

    if embedding.shape[1] != 2560:
        raise ValueError(
            f"LaMGen expects hidden dimension 2560, "
            f"but got shape {embedding.shape}"
        )

    if embedding.dtype != np.float32:
        raise ValueError(
            f"Embedding must be float32, "
            f"but got {embedding.dtype}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    print("=" * 70)
    print("[PASS] The embedding has a valid LaMGen input format.")

    return embedding


def main():
    args = parse_args()

    inspect_embedding(
        path=args.embedding
    )


if __name__ == "__main__":
    main()