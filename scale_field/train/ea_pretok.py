#!/usr/bin/env python
"""E-A pretok (评审#11 P0-1): wikitext-103 全量 → data/ea_tokens.npy, 记录 shape/max-id/sha256.

与 P2 版差异: 多 parquet shard 循环 + N_TOK 上限可配 (默认 100M, 实际取语料上限);
产出 sidecar json 供 spec 冻结引用。
先决: data/ 下已有 wt103 train parquet (curl 自 hf-mirror, 见操作单), pythia tokenizer 在 HF cache。
"""
import os, glob, json, hashlib, argparse
import numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--n_tok", type=int, default=100_000_000)
ap.add_argument("--out", default="data/ea_tokens.npy")
a = ap.parse_args()

HUB = os.path.expanduser("~/.cache/huggingface/hub")
cands = glob.glob(f"{HUB}/models--EleutherAI--pythia-*/snapshots/*/tokenizer.json")
assert cands, "未找到缓存的 pythia tokenizer.json (先下任一 pythia snapshot)"
tk = Tokenizer.from_file(cands[0]); assert tk.get_vocab_size() > 50000
print("tokenizer OK", cands[0], flush=True)

shards = sorted(glob.glob("data/wt103_train*.parquet"))
assert shards, "缺 data/wt103_train*.parquet (先 curl, 见操作单)"
buf = []
for pf in shards:
    texts = pq.read_table(pf, columns=["text"]).column("text").to_pylist()
    print(f"{pf}: rows={len(texts)}", flush=True)
    for s in texts:
        s = (s or "").strip()
        if s: buf.extend(tk.encode(s).ids)
        if len(buf) >= a.n_tok: break
    if len(buf) >= a.n_tok: break
arr = np.array(buf[:a.n_tok], dtype=np.int32)
assert arr.max() < 50304
np.save(a.out, arr)
meta = {"file": a.out, "shape": list(arr.shape), "dtype": str(arr.dtype),
        "min": int(arr.min()), "max_token_id": int(arr.max()),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shards_used": shards, "n_tok_requested": a.n_tok}
json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
print("saved", meta, flush=True)
