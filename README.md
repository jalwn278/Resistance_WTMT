# ESMFold EGFR Kinase Domain Structure Prediction Pipeline

用 Meta 的 [ESMFold](https://github.com/facebookresearch/esm)（`facebook/esmfold_v1`）预测 EGFR 野生型及耐药突变体的**激酶结构域（680–1025 aa，共 346 残基）**三维结构，用于后续耐药突变的结构影响分析。

## 项目简介

- 自动从 `all_targets.fasta` 中筛选 Header 含 `EGFR` 的序列（野生型 + 突变体）。
- 截取激酶结构域（残基 680–1025），带空序列校验。
- 半精度加载模型（默认 `bf16`）+ `model.trunk.set_chunk_size(64)` 显存切片，可在单张消费级 GPU 上稳定运行。
- 断点续传：已存在的非空 PDB 自动跳过。
- 日志输出：运行日志 + 逐条错误日志。

## 依赖安装

```bash
pip install -r requirements.txt
```

> 注意：
> - `transformers` 必须为 **4.x**（本仓库锁定 `4.40.2`）；`>=5.0` 已移除 ESMFold，`EsmForProteinFolding` 无法导入。
> - `torch` 需为带 CUDA 的构建版本（GPU 环境）；仅 CPU 也可运行但极慢。
> - 若服务器无法直连 `huggingface.co`，可用国内镜像 `export HF_ENDPOINT=https://hf-mirror.com`，或模型已缓存时设 `export HF_HUB_OFFLINE=1` 离线加载。

## 输入数据说明

- 文件：`all_targets.fasta`，放在脚本同目录下。
- 格式：标准 FASTA，Header 形如 `EGFR_P00533_T790M`（`靶点_UniProtID_突变`）。
- 脚本只处理 Header 中含 `EGFR`（不区分大小写）的序列，其余靶点自动忽略。
- 内部按 `KEYWORD`、`KD_START`、`KD_END` 三个常量控制筛选与截取，可按需修改。

## 运行命令

```bash
python predict_egfr_structure.py
```

## 输出说明

- PDB 文件输出到 `pdbs_egfr/`，命名 `EGFR_P00533_T790M.pdb` 等。
- 每个 PDB 的 **B-factor 列存储 pLDDT 置信度**（取值 `[0, 1]`），可用于结构可靠性评估。
- 运行日志：`esmfold_egfr.log`；逐条失败记录：`esmfold_egfr_errors.log`。

## 关键实现说明

- **半精度 dtype**：脚本默认使用 `bf16`（`torch.bfloat16`）。ESMFold 折叠模块在 `fp16` 下会数值溢出产生 NaN，进而触发
  `IndexError: index 0 is out of bounds for dimension 0 with size 0`；`bf16` 指数范围与 fp32 相同、数值稳定，内存占用与 fp16 一致。
  如需改用 `fp16`，修改脚本顶部 `DTYPE = torch.float16` 即可。
- **显存优化**：`model.trunk.set_chunk_size(64)` 将 folding trunk 分块计算，显著降低长序列的显存峰值。
