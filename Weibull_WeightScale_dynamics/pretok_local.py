#!/usr/bin/env python
"""Pre-tokenize wikitext-103 to a local .npy (one-time, offline-friendly).

Downloads one wikitext-103 train parquet via hf_hub_download, reads the text column with
pyarrow, tokenizes with the Pythia tokenizer (Rust `tokenizers`, loaded directly from
tokenizer.json), and saves the token ids to a .npy. The training scripts then read this .npy
without any network access.

Run: OMP_NUM_THREADS=2 python -u pretok_local.py
(set HF_ENDPOINT to a mirror if needed, e.g. HF_ENDPOINT=https://hf-mirror.com)
"""
import os, glob, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

OUT = "data"; os.makedirs(OUT, exist_ok=True)
N_TOK = 8_000_000

# --- tokenizer: 直载缓存 json (零下载, 绕 transformers bug) ---
HUB = os.path.expanduser("~/.cache/huggingface/hub")
cands = glob.glob(f"{HUB}/models--EleutherAI--pythia-*/snapshots/*/tokenizer.json")
assert cands, "未找到缓存的 pythia tokenizer.json"
tk = Tokenizer.from_file(cands[0])
assert tk.get_vocab_size() > 50000, f"tokenizer 坏 vocab={tk.get_vocab_size()}"
print("tokenizer OK", cands[0], "vocab", tk.get_vocab_size(), flush=True)

# --- 语料: 读本地已下 parquet (curl 拉好, 见 data/wt103_train0.parquet; 不再触网) ---
pf = f"{OUT}/wt103_train0.parquet"
assert os.path.exists(pf), f"缺 {pf}, 先 curl 下载"
tbl = pq.read_table(pf, columns=["text"])
texts = tbl.column("text").to_pylist()
print(f"rows={len(texts)}", flush=True)

buf = []
for s in texts:
    s = (s or "").strip()
    if s:
        buf.extend(tk.encode(s).ids)
    if len(buf) >= N_TOK:
        break
arr = np.array(buf[:N_TOK], dtype=np.int32)
np.save(f"{OUT}/wikitext_pythia_tokens.npy", arr)
assert arr.max() < 50304, f"token id {arr.max()} 超 vocab"
print(f"saved {OUT}/wikitext_pythia_tokens.npy shape={arr.shape} dtype={arr.dtype} "
      f"min={arr.min()} max={arr.max()} (< 50304 ✓)", flush=True)
