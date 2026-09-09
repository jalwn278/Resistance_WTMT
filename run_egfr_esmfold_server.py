# -*- coding: utf-8 -*-
"""
run_egfr_esmfold_server.py

在 GPU 服务器上运行 ESMFold，预测 EGFR 野生型 + 9 个突变体的
激酶结构域（Kinase Domain, 残基 680-1025, 共 346 残基）三维结构。

要点：
  * 从 all_targets.fasta 中筛选 Header 含 "EGFR" 的序列（WT + 9 突变体，共 10 条）
  * 截取激酶结构域 680-1025（1-based，含两端）；带防空串校验，避免 index 0 out of bounds
  * bf16 半精度加载模型（torch.bfloat16）节省显存 —— 不能用 fp16，否则折叠模块
    数值溢出产生 NaN，导致 compute_tm 报 "index 0 out of bounds"
  * model.trunk.set_chunk_size(64) 避免显存溢出 (OOM)
  * 结果写入 pdbs_egfr/，断点续传（已存在同名非空 .pdb 则跳过）
"""

import os
import gc
import sys

import torch
from tqdm import tqdm
from Bio import SeqIO

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

# ---------------------------------------------------------------------------
# 路径 / 运行配置
# ---------------------------------------------------------------------------
BASE_DIR = "/root/egfr"
FASTA_PATH = os.path.join(BASE_DIR, "all_targets.fasta")
OUTPUT_DIR = os.path.join(BASE_DIR, "pdbs_egfr")
ERROR_LOG = os.path.join(BASE_DIR, "esmfold_egfr_errors.log")
MODEL_NAME = "facebook/esmfold_v1"

KEYWORD = "EGFR"

# 激酶结构域范围（1-based，含两端）
KD_START = 680
KD_END = 1025

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize_filename(header):
    name = header.strip().split()[0] if header.strip() else "unknown"
    cleaned = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name)
    return cleaned or "unknown"


def convert_outputs_to_pdb(outputs):
    """把 ESMFold 输出转换为 PDB 字符串列表（每条序列一个）。"""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)

    def _np(t):
        t = t.detach().cpu()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()

    outputs_np = {k: _np(v) for k, v in outputs.items()}
    final_atom_positions = _np(final_atom_positions)
    final_atom_mask = outputs_np["atom37_atom_exists"]

    pdbs = []
    for i in range(outputs_np["aatype"].shape[0]):
        pred = Protein(
            aatype=outputs_np["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=outputs_np["residue_index"][i] + 1,
            b_factors=outputs_np["plddt"][i],
            chain_index=outputs_np["chain_index"][i] if "chain_index" in outputs_np else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def main():
    print(f"[info] device={DEVICE} | chunk_size={CHUNK_SIZE} | kinase domain {KD_START}-{KD_END}")
    if DEVICE == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)} | "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print(f"[1/4] 加载 tokenizer + 模型 {MODEL_NAME} (bf16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(
        MODEL_NAME, use_safetensors=False, torch_dtype=torch.bfloat16
    )
    model = model.to(DEVICE)
    model.eval()
    model.trunk.set_chunk_size(CHUNK_SIZE)
    print("       模型加载完成")

    print(f"[2/4] 读取 FASTA 并筛选 Header 含 '{KEYWORD}' 的序列 ...")
    records = [
        r for r in SeqIO.parse(FASTA_PATH, "fasta")
        if KEYWORD.lower() in (r.id or "").lower()
    ]
    n_wt = sum(1 for r in records if "_WT" in r.id.upper())
    print(f"       筛选到 {len(records)} 条（野生型 WT: {n_wt} | 突变体: {len(records) - n_wt}）")
    if not records:
        print("[!] 未找到 EGFR 序列，退出。")
        return

    # 截取激酶结构域 + 防空串校验
    tasks = []
    for rec in records:
        full = str(rec.seq).upper()
        if not full:
            print(f"[warn] 跳过 {rec.id}: 全长序列为空")
            continue
        kd = full[KD_START - 1: KD_END]
        if not kd:
            print(f"[warn] 跳过 {rec.id}: 截取激酶结构域后为空")
            continue
        out_path = os.path.join(OUTPUT_DIR, sanitize_filename(rec.id) + ".pdb")
        tasks.append((rec.id, kd, out_path))

    print(f"[3/4] 待预测 {len(tasks)} 条激酶结构域序列（预期 {KD_END - KD_START + 1} aa/条）")

    n_ok = n_err = n_skip = 0
    for name, kd, out_path in tqdm(tasks, desc="ESMFold", unit="seq"):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            n_skip += 1
            continue
        try:
            tokenized = tokenizer([kd], return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                outputs = model(tokenized["input_ids"].to(DEVICE))
            pdb_str = convert_outputs_to_pdb(outputs)[0]
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(pdb_str)
            os.replace(tmp, out_path)
            n_ok += 1
        except Exception as e:
            n_err += 1
            with open(ERROR_LOG, "a") as f:
                f.write(f"{name}\t{len(kd)}\t{type(e).__name__}: {e}\n")
            print(f"\n[error] {name}: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            outputs = tokenized = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"[4/4] 完成：成功 {n_ok} | 失败 {n_err} | 跳过(已存在) {n_skip}")
    if n_err:
        print(f"      失败详情见 {ERROR_LOG}")
    print(f"      输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
