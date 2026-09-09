# -*- coding: utf-8 -*-
"""
predict_egfr_structure.py —— ESMFold EGFR 激酶结构域结构预测管线

功能：
  1. 从 all_targets.fasta 中自动筛选 Header 含 "EGFR" 的序列（野生型 + 耐药突变体）
  2. 截取激酶结构域（Kinase Domain, 残基 680-1025，共 346 aa），带防空串校验
  3. 用 facebook/esmfold_v1 预测结构：
       * 半精度加载（默认 bf16）节省显存
       * model.trunk.set_chunk_size(64) 显存切片，避免 OOM
  4. 断点续传：已存在的同名非空 .pdb 自动跳过
  5. 日志输出：运行日志 esmfold_egfr.log + 逐条错误日志 esmfold_egfr_errors.log

运行：
  python predict_egfr_structure.py

重要说明（半精度 dtype 选择）：
  ESMFold 的折叠模块（folding trunk）在 fp16（torch.float16）下会数值溢出产生 NaN，
  进而在 openfold_utils 的 compute_tm 中触发
  "IndexError: index 0 is out of bounds for dimension 0 with size 0"。
  因此默认使用 bf16（torch.bfloat16）：
    * bf16 指数范围与 fp32 相同（8 位指数），数值稳定；
    * 内存占用与 fp16 相同（2 字节/参数），同样能显著节省显存。
  若你确认目标硬件/环境 fp16 可用，可将下方 DTYPE 改为 torch.float16。
"""

import gc
import logging
import os
import sys

import torch
from tqdm import tqdm
from Bio import SeqIO

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASTA_PATH = os.path.join(BASE_DIR, "all_targets.fasta")
OUTPUT_DIR = os.path.join(BASE_DIR, "pdbs_egfr")
LOG_FILE = os.path.join(BASE_DIR, "esmfold_egfr.log")
ERROR_LOG = os.path.join(BASE_DIR, "esmfold_egfr_errors.log")
MODEL_NAME = "facebook/esmfold_v1"

KEYWORD = "EGFR"        # 只处理 Header 含该关键字的序列
KD_START = 680          # 激酶结构域起始（1-based，含）
KD_END = 1025           # 激酶结构域结束（1-based，含）
CHUNK_SIZE = 64         # trunk 分块大小，避免显存溢出 (OOM)
DTYPE = torch.bfloat16  # 半精度：bf16（推荐，稳定）；fp16 会 NaN，见文件头说明
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("esmfold")


def sanitize_filename(header):
    """把 FASTA Header 转成合法的文件名（保留字母/数字/下划线/点/横线）。"""
    name = header.strip().split()[0] if header.strip() else "unknown"
    cleaned = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name)
    return cleaned or "unknown"


def convert_outputs_to_pdb(outputs):
    """把 ESMFold 输出转换为 PDB 字符串列表（每条序列一个）。

    pLDDT 置信度写入 PDB 的 B-factor 列（取值范围 [0, 1]）。
    """
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
    log.info(f"device={DEVICE} | chunk_size={CHUNK_SIZE} | 激酶结构域 {KD_START}-{KD_END}")
    if DEVICE == "cuda":
        log.info(
            "GPU: %s | 显存 %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    log.info("[1/4] 加载 tokenizer + 模型 %s (半精度) ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(
        MODEL_NAME, use_safetensors=False, torch_dtype=DTYPE
    )
    model = model.to(DEVICE)
    model.eval()
    model.trunk.set_chunk_size(CHUNK_SIZE)
    log.info("       模型加载完成")

    log.info("[2/4] 读取 FASTA 并筛选 Header 含 '%s' 的序列 ...", KEYWORD)
    records = [
        r for r in SeqIO.parse(FASTA_PATH, "fasta")
        if KEYWORD.lower() in (r.id or "").lower()
    ]
    n_wt = sum(1 for r in records if "_WT" in r.id.upper())
    log.info("       筛选到 %d 条（野生型 WT: %d | 突变体: %d）",
             len(records), n_wt, len(records) - n_wt)
    if not records:
        log.error("未找到含 '%s' 的序列，退出。", KEYWORD)
        return

    # 截取激酶结构域 + 防空串校验（避免 index 0 out of bounds）
    tasks = []
    for rec in records:
        full = str(rec.seq).upper()
        if not full:
            log.warning("跳过 %s：全长序列为空", rec.id)
            continue
        kd = full[KD_START - 1: KD_END]
        if not kd:
            log.warning("跳过 %s：截取激酶结构域后为空", rec.id)
            continue
        out_path = os.path.join(OUTPUT_DIR, sanitize_filename(rec.id) + ".pdb")
        tasks.append((rec.id, kd, out_path))

    log.info("[3/4] 待预测 %d 条激酶结构域序列（预期 %d aa/条）",
             len(tasks), KD_END - KD_START + 1)

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
            log.error("%s: %s: %s", name, type(e).__name__, e)
        finally:
            outputs = tokenized = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    log.info("[4/4] 完成：成功 %d | 失败 %d | 跳过(已存在) %d", n_ok, n_err, n_skip)
    if n_err:
        log.info("      失败详情见 %s", ERROR_LOG)
    log.info("      输出目录: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
