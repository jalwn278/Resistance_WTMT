from pathlib import Path
import argparse
import torch
import gc
import numpy as np

from transformers import AutoTokenizer, AutoModel

def read_fasta(fasta_path):
    allowed_suffixes = {'.fasta', '.fa', '.faa'}
    path = Path(fasta_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File doesn't exist: {path.resolve()}")

    if not path.is_file():
        raise IsADirectoryError(f"Fasta path is not a file: {path.resolve()}")

    if path.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"Unsupported fasta suffix: {path.suffix!r}. "
                         f"Allowed suffixes: {sorted(allowed_suffixes)}")

    header = None
    sequence_lines = []
    with path.open('r', encoding = 'utf-8') as handle:
        for line_numbers, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if header is not None:
                    raise ValueError(f"Multiple Fasta records found in {path.resolve()} "
                                     f"Second header appears at line {line_numbers}")

                if not line[1:].strip():
                    raise ValueError(f"Empty Fasta header in {path.resolve()} "
                                     f"at line {line_numbers}")

                header = line[1:].strip()
                continue

            if header is None:
                raise ValueError(f"Protein sequence appears before Fasta header "
                                f"in {path.resolve()} at line {line_numbers}")

            cleaned_line = "".join(line.split()).upper()
            if cleaned_line:
                sequence_lines.append(cleaned_line)

    if header is None:
        raise ValueError(f"No Fasta header was found in: {path.resolve()}")

    sequence = ''.join(sequence_lines)
    if not sequence:
        raise ValueError(f"Fasta sequence is empty: {path.resolve()}")

    return header, sequence

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--wt-fasta",
            required = True,
            help = '...'
    )

    parser.add_argument(
            "--mt-fasta",#mt_fasta
            required = True,
            help = '...'
    )

    parser.add_argument(
            "--model-id",
            default="biohub/ESMC-6B",
            help="Hugging Face model ID for ESM-C."
    )

    parser.add_argument(
            "--output-dir",
            default="resistance_project/embeddings",
            help="Directory used to save WT and MT .npy embeddings."
    )

    parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing .npy files.",
    )

    return parser.parse_args()
    #Namespace(wt_fasta="data/P00533_WT.fasta",mt_fasta="data/P00533_T790M.fasta")

def validate_sequence(sequence, label):
    label = label.upper()
    if label not in {"WT", "MT"}:
        raise ValueError(f"Unknown label: {label!r}")

    allowed_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    sequence_characters = set(sequence)
    invalid_characters = sequence_characters - allowed_amino_acids

    if invalid_characters:
        raise ValueError(f"Invalid amino acid characters in {label} sequence "
                         f"{sorted(invalid_characters)}")

def patch_esmc_rotary_meta_buffer():
    from transformers.models.esmc.modeling_esmc import(RotaryEmbedding,)

    if getattr(
        RotaryEmbedding,
        '_meta_buffer_patch_applied',
        False,
    ):
        return

    original_reset_parameters = (RotaryEmbedding.reset_parameters)

    def safe_reset_parameters(self, device=None):
        if(device is not None and torch.device(device).type == 'meta'):
            device = torch.device('cpu')

        return original_reset_parameters(self,device=device)

    RotaryEmbedding.reset_parameters = (safe_reset_parameters)

    RotaryEmbedding._meta_buffer_patch_applied = True

    print('[INFO] Applied ESM-C RoPE '
          'meta-buffer compatibility patch')


def load_esmc(model_id):
    print(f"loading ESM-C model: {model_id}")
    patch_esmc_rotary_meta_buffer()

    revision = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"#固定revision

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
    ).eval()

    return tokenizer, model

def extract_embedding(model, tokenizer, sequence, label):
    encoded = tokenizer(
        sequence,
        return_tensors='pt',#pytorch
        padding=False,#no <pad>
        truncation=False,#No truncation保留完整输入
        return_special_tokens_mask=True,#find <cls> <eos>
    )

    label = label.upper()
    if label not in {"WT", "MT"}:
        raise ValueError(
            f"Unknown label: {label!r}"
        )

    #去掉specialtoken 只处理protein
    special_tokens_mask = encoded.pop('special_tokens_mask')

    attention_mask = encoded['attention_mask']
    input_ids = encoded["input_ids"]
    #input_ids's format[batch_size, token_count] batch_size为1 一次只处理一个
    if input_ids.ndim != 2:
        raise ValueError("Expected for [batch_size, token_count]")

    if input_ids.shape[0] != 1:
        raise ValueError(f"Batch size is {input_ids.shape[0]} and not 1")


    token_count = encoded["input_ids"].shape[1]
    max_tokens = getattr(model.config,
                        "max_position_embeddings",
                        2048,
    )
    if max_tokens is None:
        max_tokens = 2048
    if max_tokens is not None and token_count > max_tokens:
        raise ValueError(f"{token_count} tokens, model limit is {max_tokens}")

    model_inputs = {
        name: tensor.to(model.device)
        for name, tensor in encoded.items()
    }

    #no grad 不反向传播
    with torch.inference_mode():
        outputs = model(**model_inputs)

    hidden_state = getattr(outputs,
                           "last_hidden_state",
                           None,
    )
    if hidden_state is None:
        raise RuntimeError("Esm-c outputs doesn't contain last_hidden_state")
    if hidden_state.ndim != 3:
        raise ValueError("Expected for [batch_szie, token_count, hidden_dimension] "
                         f"but got shape {tuple(hidden_state.shape)}")
    if hidden_state.shape[1] != token_count:
        raise ValueError(f"input_ids = {token_count} "
                         f"hidden_state = {hidden_state.shape[1]}")
    if hidden_state.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1, "
            f"but got hidden-state shape {tuple(hidden_state.shape)}"
        )

    token_embedding = hidden_state[0]

    #不是specialtoken且是有效input
    residue_mask = (
        special_tokens_mask[0].eq(0)
        & attention_mask[0].eq(1)
    )
    #将mask写在token_embedding的设备上 并保留mask为True的行
    residue_mask = residue_mask.to(token_embedding.device)
    embedding= token_embedding[residue_mask]

    embedding = (
        embedding.detach().to(
            device = 'cpu',
            dtype = torch.float32,
        )
        .contiguous()
    )

    if embedding.shape[0] != len(sequence):
        raise ValueError(
            f"{label} embeeding doesn't match sequence length: "
            f"sequence={len(sequence)} "
            f"embedding={embedding.shape[0]} "
            f"token_count={token_count}"
        )
    if embedding.ndim != 2:
        raise ValueError(
            f"{label} embeddingmust be 2-dimensional "
            f"but got shape {tuple(embedding.shape)}"
        )

    model_hidden_size = getattr(
        model.config,
        'hidden_size',
        None,
    )
    if model_hidden_size is None:
        model_hidden_size = getattr(
            model.config,
            'd_model',
            None,
        )
    if (model_hidden_size is not None and embedding.shape[1] != model_hidden_size):
        raise ValueError(
            f"{label} embedding doesn't match "
            f"config={model_hidden_size} "
            f"embedding={embedding.shape[1]}"
        )

    if embedding.shape[1] != 2560:
        raise ValueError(
            "LaMGen expects protein embedding is 2560-dimensional "
            f"but {label} embedding shape is {tuple(embedding.shape)}"
        )

    if not torch.isfinite(embedding).all().item():
        raise ValueError(
            f"{label} embedding contains NaN of infinite values"
        )

    return embedding

def save_embedding(embedding, output_path, label, force=False):
    label = label.upper()
    if label not in {"WT", "MT"}:
        raise ValueError(
            f"Unknown label: {label!r}"
        )

    if not isinstance(embedding, torch.Tensor):
        raise TypeError(
            f"{label} embedding must be a torch.Tensor "
            f"but got {type(embedding).__name__}"
        )
    if embedding.ndim != 2:
        raise ValueError(
            f"{label} embedding must be 2-dimensional, "
            f"but got shape {tuple(embedding.shape)}"
        )
    if embedding.shape[1] != 2560:
        raise ValueError(
            f"{label} embedding must have hidden dimension 2560, "
            f"but got shape {tuple(embedding.shape)}"
        )

    if embedding.device.type != 'cpu':
        raise ValueError(
            f"{label} embedding must be on CPU before saving "
            f"but got device {embedding.device}"
        )
    if embedding.dtype != torch.float32:
        raise ValueError(
            f"{label} embedding must be torch.float32 "
            f"but got {embedding.dtype}"
        )
    if not torch.isfinite(embedding).all().item():
        raise ValueError(
            f"{label} embedding contains NaN or infinite values"
        )

    path = Path(output_path).expanduser()
    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"{label} output path is a directory "
            f"{path.resolve()}"
        )
    if path.suffix == "":
        path = path.with_suffix(".npy")
    elif path.suffix.lower() != ".npy":
        raise ValueError(
            f"{label} output file must use the .npy suffix "
            f"but got: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {path.resolve()} "
            "Use --force to overwrite it"
        )

    embedding_array = (embedding.detach().contiguous().numpy())

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(
                handle,
                embedding_array,
                allow_pickle=False,
            )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    loaded = np.load(
        path,
        allow_pickle=False,
    )
    if loaded.shape != embedding_array.shape:
        raise RuntimeError(
            f"{label} shape changed after saving "
            f"before={embedding_array.shape} "
            f"after={loaded.shape}"
        )
    if loaded.dtype != np.float32:
        raise RuntimeError(
            f"{label} saved dtype is not float32: "
            f"{loaded.dtype}"
        )
    if not np.isfinite(loaded).all():
        raise RuntimeError(
            f"Saved {label} embedding contains NaN or Inf"
        )

    if not np.array_equal(
        loaded,
        embedding_array
    ):
        raise RuntimeError(f"{label} embedding values changed after saving")

    print(
        f"[PASS] Saved {label} embedding\n"
        f"path: {path.resolve()}\n"
        f"shape: {loaded.shape}\n"
        f"dtype: {loaded.dtype}"
    )

    return path

def main():
    args = parse_args()
    wt_header, wt_sequence = read_fasta(args.wt_fasta)
    mt_header, mt_sequence = read_fasta(args.mt_fasta)
    validate_sequence(wt_sequence, "WT")
    validate_sequence(mt_sequence, "MT")

    print("WT/MT FASTA information")
    print("=" * 70)
    print(f"WT header : {wt_header}")
    print(f"WT length : {len(wt_sequence)}")
    print(f"MT header : {mt_header}")
    print(f"MT length : {len(mt_sequence)}")

    output_dir = Path(args.output_dir).expanduser()
    wt_name = Path(args.wt_fasta).stem
    mt_name = Path(args.mt_fasta).stem
    wt_output_path = output_dir / f"{wt_name}.npy"
    mt_output_path = output_dir / f"{mt_name}.npy"

    print("Output paths")
    print("=" * 70)
    print(f"WT output : {wt_output_path.resolve()}")
    print(f"MT output : {mt_output_path.resolve()}")

    print("=" * 70)
    print(f"Loading ESM-C model: {args.model_id}")

    tokenizer, model = load_esmc(args.model_id)

    print("[INFO] Extracting WT embedding...")
    wt_embedding = extract_embedding(
    sequence=wt_sequence,
    tokenizer=tokenizer,
    model=model,
    label="WT",
    )
    print(
        f"[PASS] WT embedding shape: "
        f"{tuple(wt_embedding.shape)}"
    )
    wt_saved_path = save_embedding(
            embedding=wt_embedding,
            output_path=wt_output_path,
            label="WT",
            force=args.force,
    )
    del wt_embedding
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[INFO] Extracting MT embedding...")
    mt_embedding = extract_embedding(
        sequence=mt_sequence,
        tokenizer=tokenizer,
        model=model,
        label="MT",
    )
    print(
        f"[PASS] MT embedding shape: "
        f"{tuple(mt_embedding.shape)}"
    )
    mt_saved_path = save_embedding(
        embedding=mt_embedding,
        output_path=mt_output_path,
        label="MT",
        force=args.force,
    )

    print("[PASS] All embeddings were generated successfully")
    print("=" * 70)
    print(f"WT embedding : {wt_saved_path.resolve()}")
    print(f"MT embedding : {mt_saved_path.resolve()}")

if __name__ == "__main__":
    main()
