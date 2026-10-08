from pathlib import Path
from datetime import datetime, timezone
from transformers import GPT2Config
import argparse
import csv
import hashlib
import numpy as np
import torch.nn.functional as F
import torch
import random

import sys

script_path = Path(__file__).resolve()
lamgen_root = script_path.parents[2]

if str(lamgen_root) not in sys.path:
    sys.path.insert(0, str(lamgen_root))

from utils.bert_tokenizer import ExpressionBertTokenizer
from model.lamgen_model import LaMGen_dual

def parse_args():
    script_path = Path(__file__).resolve()
    resistance_root = script_path.parents[1]
    lamgen_root = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description = (
            "Generate molecule candidates conditioned on WT and mutant protein embedding"
        )
    )

    parser.add_argument(
        "--wt-embedding",
        type = Path,
        required = True,
        help = "Path to the WT protein embedding .npy file"
    )

    parser.add_argument(
        "--mt-embedding",
        type = Path,
        required = True,
        help = "Path to the MT protein embedding .npy file"
    )

    parser.add_argument(
        "--pretrain-path",
        type = Path,
        default = lamgen_root/"Pretrained_model",
        help = "Path to the pretrained molecular GPT model directory",
    )

    parser.add_argument(
        "--model-path",
        type = Path,
        default = (lamgen_root/"checkpoint"/"dual"/"dual_target_ckpt"),
        help = 'Path to the LaMGen dual-target checkpoint',
    )

    parser.add_argument(
        "--vocab-path",
        type = Path,
        default = lamgen_root/"data"/"torsion_voc.csv",
        help = "Path to the LaMGen molecular token vocabulary",
    )

    parser.add_argument(
        "--batch-size",
        type = int,
        default = 1,
        help = "Number of molecular sequences generated in one batch",
    )

    parser.add_argument(
        "--epochs",
        type = int,
        default = 1,
        help = "Number of repeated generation rounds"
    )

    parser.add_argument(
        "--max-length",
        type = int,
        default = 195,
        help = "Maximum number of autoregressive generation steps",
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 0,
        help = "Base random seed used for molecular sampling",
    )

    parser.add_argument(
        "--output-dir",
        type = Path,
        default = (resistance_root/"generation/"),
        help = "Root directory used to organize generation results by WT/MT pair",
    )

    parser.add_argument(
        "--target1",
        type = str,
        default = None,
        help = "Name written to the target1 CSV column",
    )

    parser.add_argument(
        "--target2",
        type = str,
        default = None,
        help = "Name written to the target2 CSV column",
    )

    parser.add_argument(
        "--force-overwrite",
        action = "store_true",
        help = (
            "Overwrite the existing candidates CSV when "
            "the same WT/MT pair already exists."),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than 0."
        )

    if args.epochs <= 0:
        parser.error(
            "--epochs must be greater than 0."
        )

    if args.max_length <= 0:
        parser.error(
            "--max-length must be greater than 0."
        )

    if args.seed < 0:
        parser.error(
            "--seed must be greater than or equal to 0."
        )

    if args.target1 is None:
        args.target1 = args.wt_embedding.stem

    if args.target2 is None:
        args.target2 = args.mt_embedding.stem

    return args

def load_embedding(path, label)->tuple[Path, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} embedding file was not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(
            f"{label} embedding path is not a file: {path}"
        )
    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"{label} embedding must use the .npy suffix, "
            f"but got {path.suffix!r}."
        )

    embedding = np.load(path, allow_pickle=False)
    if embedding.ndim != 2:
        raise ValueError(
            f"{label} embedding must be two-dimensional"
        )
    if embedding.shape[0] <= 0:
        raise ValueError(
            f"{label} embedding contains no residue rows."
        )

    if embedding.shape[1] != 2560:
        raise ValueError(
            f"{label} embedding must have hidden dimension 2560, "
            f"but got shape {embedding.shape}."
        )

    if embedding.dtype != np.float32:
        raise ValueError(
            f"{label} embedding must use float32, "
            f"but got {embedding.dtype}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"{label} embedding contains NaN or infinite values."
        )

    return path, embedding

def compute_embedding_hash(embedding):
    if not isinstance(embedding, np.ndarray):
        raise TypeError(
            "embedding must be a Numpy ndarray "
            f"but got {type(embedding).__name__}"
        )

    embedding = np.ascontiguousarray(embedding)

    return hashlib.sha256(
        embedding.tobytes(order='C')
    ).hexdigest()

def build_pair_id(wt_hash, mt_hash):
    wt_hash = wt_hash.strip().lower()
    mt_hash = mt_hash.strip().lower()

    if len(wt_hash) != 64:
        raise ValueError(
            "wt_hash must be a 64-character SHA-256 hash"
        )
    if len(mt_hash) != 64:
        raise ValueError(
            "mt_hash must be a 64-character SHA-256 hash."
        )

    pair_text = (
        f"WT:{wt_hash}\n"
        f"MT:{mt_hash}")

    full_pair_hash = hashlib.sha256(
        pair_text.encode("utf-8")
    ).hexdigest()

    return full_pair_hash[:16]

def resolve_output_paths(output_dir,pair_id,):
    pair_id = pair_id.strip()

    output_dir = Path(output_dir).expanduser().resolve()
    pair_dir = (output_dir / pair_id)
    pair_dir.mkdir(parents=True,exist_ok=True,)
    csv_path = (pair_dir / "candidates.csv")

    return pair_dir, csv_path

def decide_csv_action(csv_path, force_overwrite):
    csv_path = Path(
        csv_path
    ).expanduser().resolve()

    if not csv_path.exists():
        return "create"
    if not csv_path.is_file():
        raise IsADirectoryError(
            "Candidate CSV path exists but is not a file: "
            f"{csv_path}"
        )
    if not force_overwrite:
        return "skip"

    return "overwrite"

def padding_embedding_lengths(wt_embedding, mt_embedding)->tuple[np.ndarray,np.ndarray]:
    if wt_embedding.shape[1] != mt_embedding.shape[1]:
        raise ValueError(
            "WT and MT embeddings must have the same "
            "hidden dimension, "
            f"but got {wt_embedding.shape[1]} and {mt_embedding.shape[1]}."
        )

    max_length = max(wt_embedding.shape[0], mt_embedding.shape[0])

    padding_wt_embedding = np.pad(
        wt_embedding,
        (
            (0, max_length-wt_embedding.shape[0]), (0,0)
        ),
        mode = 'constant'
    )
    padding_mt_embedding = np.pad(
        mt_embedding,
        (
            (0, max_length-mt_embedding.shape[0]), (0,0)
        ),
        mode = 'constant'
    )

    return (padding_wt_embedding, padding_mt_embedding)

def prepare_protein_batches(protein1, protein2, batch_size, device):
    protein_batch1 = (
        torch.tensor(
            protein1, dtype = torch.float32
        )
        .to(device)
        .repeat(
            batch_size,1,1
        )
    )

    protein_batch2 = (
        torch.tensor(
            protein2, dtype = torch.float32
        )
        .to(device)
        .repeat(
            batch_size,1,1
        )
    )

    return (protein_batch1, protein_batch2)

def load_tokenizer(vocab_path):
    vocab_path = Path(vocab_path).expanduser().resolve()

    if not vocab_path.exists():
        raise FileNotFoundError(
            "Tokenizer vocabulary was not found: "
            f"{vocab_path}"
        )

    if not vocab_path.is_file():
        raise IsADirectoryError(
            "Tokenizer vocabulary path is not a file: "
            f"{vocab_path}"
        )

    tokenizer = (ExpressionBertTokenizer.from_pretrained(str(vocab_path)))

    return tokenizer

def lamgen_config():
    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        model_type="GPT2LMHeadModel",
        vocab_size=836,
        n_positions=1800,
        n_ctx=380,
        n_embd=768,
        n_layer=12,
        n_head=8,
        task_specific_params={
            "text-generation":{
                "do_sample":True,
                "max_length":500
            }
        }
    )

    return config

def load_lamgen_dual_model(pretrain_path, model_path, config, device):
    pretrain_path = Path(
        pretrain_path
    ).expanduser().resolve()

    model_path = Path(
        model_path
    ).expanduser().resolve()

    model = LaMGen_dual(
        pretrain_path=str(pretrain_path),
        config=config
    )

    param_dict = {
        key.replace("module.",""):value
        for key, value in torch.load(
                str(model_path),
                map_location=device
            ).items()
    }

    model.load_state_dict(
        param_dict
    )

    return model

def prepare_generation_inputs(
        tokenizer,batch_size,
        text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    input_ids = []
    input_ids.extend(
        tokenizer.encode(
            text, add_special_tokens=False
        )
    )

    input_length = len(input_ids)
    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)

    return input_tensor

@torch.no_grad()
def generate_molecules(model,tokenizer,input_tensor,protein_batch1,protein_batch2,max_length,seed,device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    model.eval()

    seq_list = []
    finished = torch.zeros(
        input_tensor.shape[0],
        1
    ).byte().to(device)

    eos_ids = tokenizer.encode("<|endofmask|>",add_special_tokens=False)
    if len(eos_ids) != 1:
        raise ValueError(
            "<|endofmask|> must correspond to exactly one token."
        )
    eos_id = eos_ids[0]

    for i in range(max_length):
        inputs = input_tensor.to(device)
        outputs = model(inputs, protein_batch1, protein_batch2)

        logits = outputs.logits
        logits = F.softmax(logits[:,-1,:],dim=-1)
        last_token_id = torch.multinomial(logits,1)

        EOS_sampled = (last_token_id == eos_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print("End")
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to("cpu")),dim=1)
        seq_list.append(last_token)

    seq_list = np.array(seq_list).T

    return seq_list

def decode(matrix):
    chars = []

    for i in matrix:
        if i == "<|endofmask|>":
            break
        chars.append(i)

    seq = " ".join(chars)

    return seq

def split_sequence(sequence):
    tokens = sequence.split()
    if "GEO" not in tokens:
        return "".join(tokens), ""

    geo_index = tokens.index("GEO")
    smiles_tokens = tokens[:geo_index]
    geo_tokens = tokens[geo_index + 1:]

    smiles = "".join(smiles_tokens)
    geo = " ".join(geo_tokens)

    return smiles, geo

def save_candidates_csv(csv_path, candidate_rows, csv_action):
    csv_path = Path(csv_path).expanduser().resolve()
    if csv_action not in {
        "create",
        "overwrite",
    }:
        raise ValueError(
            f"Unsupported CSV action: {csv_action}"
        )

    temp_path = csv_path.with_suffix(".tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Smiles","GEO","target1","target2",])
            writer.writerows(candidate_rows)
        temp_path.replace(csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def main():
    args = parse_args()

    wt_path, wt_embedding = load_embedding(path=args.wt_embedding,label="WT")
    mt_path, mt_embedding = load_embedding(path=args.mt_embedding,label="MT")

    wt_hash = compute_embedding_hash(wt_embedding)
    mt_hash = compute_embedding_hash(mt_embedding)

    pair_id = build_pair_id(wt_hash = wt_hash,mt_hash = mt_hash)
    pair_dir, csv_path = resolve_output_paths(output_dir=args.output_dir,pair_id=pair_id)

    csv_action = decide_csv_action(
        csv_path = csv_path,
        force_overwrite = args.force_overwrite
    )
    if csv_action == "skip":
        print(
            f"[SKIP] Candidate CSV already exists: "
            f"{csv_path}"
        )
        return
    print(
        f"[INFO] CSV action: {csv_action}"
    )

    padding_wt_embedding, padding_mt_embedding = (
        padding_embedding_lengths(
            wt_embedding = wt_embedding,
            mt_embedding = mt_embedding
        )
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    protein_batch1, protein_batch2 = (
    prepare_protein_batches(
        protein1=padding_wt_embedding,
        protein2=padding_mt_embedding,
        batch_size=args.batch_size,
        device=device,
        )
    )

    tokenizer = load_tokenizer(vocab_path=args.vocab_path)
    config = lamgen_config()
    model = load_lamgen_dual_model(
        pretrain_path=args.pretrain_path,
        model_path=args.model_path,
        config=config,
        device=device
    )

    seq_all = []
    for epoch in range(args.epochs):
        epoch_seed = args.seed + epoch
        input_tensor = prepare_generation_inputs(tokenizer=tokenizer, batch_size = args.batch_size)

        seq_list = generate_molecules(model=model,tokenizer=tokenizer,input_tensor=input_tensor,protein_batch1=protein_batch1,protein_batch2=protein_batch2,max_length=args.max_length,seed=epoch_seed,device=device)
        seq_all.extend(seq_list)

    decoded_sequences = []
    for part in seq_all:
        decoded_seq = decode(part)
        decoded_sequences.append(decoded_seq)

    candidate_rows = []
    for decoded_seq in decoded_sequences:
        smiles, geo = split_sequence(decoded_seq)

        candidate_rows.append([smiles,geo,args.target1,args.target2])

    save_candidates_csv(csv_path=csv_path, candidate_rows=candidate_rows, csv_action=csv_action)

if __name__ == "__main__":
    main()
