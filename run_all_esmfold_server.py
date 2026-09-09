# -*- coding: utf-8 -*-
"""
run_all_esmfold_server.py

在 GPU 服务器上对 all_targets.fasta 中全部 206 条序列（EGFR WT + 全部耐药突变体，
及另外 44 个靶点）做 ESMFold 结构预测。

结构域策略（与 EGFR 激酶域 680-1025 同一思路）：
  * 全长 <= 1024 的序列 —— 直接预测全长
  * 全长 > 1024 的序列 —— 只折叠「突变位点附近的催化/功能结构域」，
    保证所有耐药突变位点都落在被折叠的片段内（经本地校验，0 个突变位点落在域外）

要点（沿用 EGFR 已跑通的配置）：
  * bf16（torch.bfloat16）加载模型 —— 不能用 fp16，否则折叠模块数值溢出产生 NaN，
    导致 compute_tm 报 "index 0 out of bounds"
  * model.trunk.set_chunk_size(64) 避免显存溢出
  * 断点续传：已存在同名非空 .pdb 则跳过
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
OUTPUT_DIR = os.path.join(BASE_DIR, "pdbs_all")
ERROR_LOG = os.path.join(BASE_DIR, "esmfold_all_errors.log")
DOMAIN_LOG = os.path.join(BASE_DIR, "esmfold_all_domains.tsv")
MODEL_NAME = "facebook/esmfold_v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 64

# 催化/功能结构域（1-based，含两端）。仅列出全长 > 1024 的靶点；其余跑全长。
# 描述中标注了覆盖的耐药突变位点，保证截取片段完整包含它们。
DOMAIN_MAP = {
    "EGFR":     (680, 1025, "kinase domain (T790/C797/D761/L747/L718/L844/G719/L792/G796)"),
    "ABL1":     (242, 493,  "kinase domain (M244/G250/Y253/E255/T315/F317/M351/H396)"),
    "ALK":      (1116, 1392, "kinase domain (C1156/I1171/F1174/L1196/G1202/S1206/G1269)"),
    "PDGFRA":   (593, 935,  "kinase domain (D842)"),
    "ROS1":     (1945, 2222, "kinase domain (F2004/L2026/G2032/D2033)"),
    "RET":      (713, 1012, "kinase domain (V804/G810)"),
    "MET":      (1078, 1337, "kinase domain (F1200/D1228/Y1230)"),
    "ERBB2":    (720, 987,  "kinase domain (L755/T798)"),
    "PIK3CA":   (330, 1068, "catalytic core C2+helical+kinase (C420/E545/H1047)"),
    "MTOR":     (2015, 2516, "FRB + kinase domain (S2035/F2038)"),
    "JAK2":     (380, 809,  "SH2-like + JH2 pseudokinase (D420/E592)"),
    "SF3B1":    (600, 1304, "C-terminal HEAT-repeat window (R1074)"),
    "TOP2A":    (1, 1024,   "N-terminal ATPase+TOPRIM+core (R487)"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def target_of(header):
    """Header 形如 TARGET_UNIPROT_MUTATION，取 TARGET 部分（可能含下划线）。"""
    parts = header.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 else (parts[0] if parts else "unknown")


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


def build_tasks(records):
    """把每条序列映射成 (header, fold_seq, out_path, domain_desc)。"""
    tasks = []
    domain_rows = []
    for rec in records:
        full = str(rec.seq).upper()
        if not full:
            print(f"[warn] 跳过 {rec.id}: 全长序列为空")
            continue
        target = target_of(rec.id)
        if target in DOMAIN_MAP:
            s, e, desc = DOMAIN_MAP[target]
            fold = full[s - 1: e]
        else:
            s, e = 1, len(full)
            fold = full
            desc = "full-length"
        if not fold:
            print(f"[warn] 跳过 {rec.id}: 截取后为空")
            continue
        out_path = os.path.join(OUTPUT_DIR, sanitize_filename(rec.id) + ".pdb")
        tasks.append((rec.id, fold, out_path))
        domain_rows.append((rec.id, target, len(full), s, e, len(fold), desc))
    return tasks, domain_rows


def main():
    print(f"[info] device={DEVICE} | chunk_size={CHUNK_SIZE}")
    if DEVICE == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)} | "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print(f"[1/5] 加载 tokenizer + 模型 {MODEL_NAME} (bf16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(
        MODEL_NAME, use_safetensors=False, torch_dtype=torch.bfloat16
    )
    model = model.to(DEVICE)
    model.eval()
    model.trunk.set_chunk_size(CHUNK_SIZE)
    print("       模型加载完成")

    print(f"[2/5] 读取 FASTA {FASTA_PATH} ...")
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"       共 {len(records)} 条序列")

    tasks, domain_rows = build_tasks(records)
    with open(DOMAIN_LOG, "w") as f:
        f.write("header\ttarget\tfull_len\tstart\tend\tfold_len\tdomain\n")
        for row in domain_rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"[3/5] 待预测 {len(tasks)} 条（结构域映射见 {DOMAIN_LOG}）")

    n_ok = n_err = n_skip = 0
    for name, fold, out_path in tqdm(tasks, desc="ESMFold", unit="seq"):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            n_skip += 1
            continue
        try:
            tokenized = tokenizer([fold], return_tensors="pt", add_special_tokens=False)
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
                f.write(f"{name}\t{len(fold)}\t{type(e).__name__}: {e}\n")
            print(f"\n[error] {name}: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            outputs = tokenized = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"[5/5] 完成：成功 {n_ok} | 失败 {n_err} | 跳过(已存在) {n_skip}")
    if n_err:
        print(f"      失败详情见 {ERROR_LOG}")
    print(f"      输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
