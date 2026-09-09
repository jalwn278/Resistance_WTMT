# -*- coding: utf-8 -*-
import traceback
import torch
from Bio import SeqIO
from transformers import EsmForProteinFolding, AutoTokenizer

FASTA = "/root/egfr/all_targets.fasta"
KD_START, KD_END = 680, 1025

recs = [r for r in SeqIO.parse(FASTA, "fasta") if "egfr" in r.id.lower()]
rec = recs[0]
seq = str(rec.seq).upper()[KD_START - 1: KD_END]
print("name:", rec.id, "len:", len(seq))

tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", use_safetensors=False, torch_dtype=torch.bfloat16
).cuda().eval()
model.trunk.set_chunk_size(64)

inp = tok([seq], return_tensors="pt", add_special_tokens=False)
print("input_ids shape:", inp["input_ids"].shape)

with torch.no_grad():
    out = model(inp["input_ids"].cuda())

print("=== output keys + shapes ===")
for k, v in out.items():
    try:
        shp = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(f"  {k}: {shp}")
    except Exception as e:
        print(f"  {k}: <{type(v).__name__}> {e}")

print("=== trying convert ===")
try:
    from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    final = atom14_to_atom37(out["positions"][-1], out)
    print("atom14_to_atom37 OK, final shape:", tuple(final.shape))
    def _np(t):
        t = t.detach().cpu()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()
    o = {k: _np(v) for k, v in out.items()}
    pred = Protein(
        aatype=o["aatype"][0],
        atom_positions=_np(final)[0],
        atom_mask=o["atom37_atom_exists"][0],
        residue_index=o["residue_index"][0] + 1,
        b_factors=o["plddt"][0],
        chain_index=o["chain_index"][0] if "chain_index" in o else None,
    )
    pdb_str = to_pdb(pred)
    print("to_pdb OK, length:", len(pdb_str))
    with open("/root/egfr/debug_one.pdb", "w") as f:
        f.write(pdb_str)
    print("WROTE /root/egfr/debug_one.pdb")
except Exception:
    traceback.print_exc()
