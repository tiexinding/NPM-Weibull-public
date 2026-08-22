# -*- coding: utf-8 -*-
"""
  python p4_blimp_eval.py --runs <dir> [--runs <dir2> ...] --data blimp_data/blimp-master/data \
      --tok blimp_data/pythia_tokenizer --out data/blimp_s0.json [--steps 8000] [--bs 64]
"""
import json, glob, os, argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', action='append', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--tok', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--subsets', default='all', help='all or comma-separated subset names (for smoke tests)')
ap.add_argument('--pairs_per_subset', type=int, default=1000)
a = ap.parse_args()

print('[1] torch/transformers...', flush=True)
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'    device={dev}', flush=True)

tok = AutoTokenizer.from_pretrained(a.tok)
tok.pad_token = tok.eos_token
cfg = GPTNeoXConfig(vocab_size=50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                    intermediate_size=2048, rotary_pct=0.25, rotary_emb_base=10000,
                    max_position_embeddings=2048, use_parallel_residual=True,
                    layer_norm_eps=1e-5, hidden_act='gelu')

#
files = sorted(glob.glob(f'{a.data}/*.jsonl'))
if a.subsets != 'all':
    keep = set(a.subsets.split(','))
    files = [f for f in files if Path(f).stem in keep]
subsets = {}
for f in files:
    rows = [json.loads(l) for l in open(f)][:a.pairs_per_subset]
    subsets[Path(f).stem] = [(r['sentence_good'], r['sentence_bad']) for r in rows]
print(f'[2] BLiMP: {len(subsets)} subsets, {sum(len(v) for v in subsets.values())} pairs', flush=True)

@torch.no_grad()
def sent_logprobs(model, sents):
    """"""
    out = np.zeros(len(sents))
    for i in range(0, len(sents), a.bs):
        chunk = sents[i:i + a.bs]
        enc = tok(chunk, return_tensors='pt', padding=True)
        ids = enc.input_ids.to(dev); att = enc.attention_mask.to(dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tgt = ids[:, 1:]
        gather = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [B, L-1]
        mask = att[:, 1:].float()
        out[i:i + a.bs] = (gather * mask).sum(-1).cpu().numpy()
    return out

#
results = json.load(open(a.out)) if os.path.exists(a.out) else {}
run_dirs = []
for root in a.runs:
    run_dirs += sorted(glob.glob(f'{root}/p4grid_*/'))
for d in run_dirs:
    name = d.rstrip('/').split('/')[-1]
    tag = f"{Path(d.rstrip('/')).parent.name}/{name}"
    if tag in results:
        continue
    ck = f'{d}W_ckpts/step{a.steps}.pt'
    if not os.path.exists(ck):
        print(f'  [skip no ckpt] {tag}', flush=True)
        continue
    sd = torch.load(ck, map_location='cpu', weights_only=True)
    model = GPTNeoXForCausalLM(cfg)
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.to(dev).eval()
    accs = {}
    for sub, pairs in subsets.items():
        lg = sent_logprobs(model, [p[0] for p in pairs])
        lb = sent_logprobs(model, [p[1] for p in pairs])
        accs[sub] = float((lg > lb).mean())
    overall = float(np.mean(list(accs.values())))
    results[tag] = dict(overall=overall, n_subsets=len(accs), subsets=accs,
                        miss=len(miss), unexp=len(unexp))
    json.dump(results, open(a.out, 'w'), indent=1)
    print(f'  {tag}: BLiMP overall = {overall:.4f}', flush=True)
    del model, sd
    if dev == 'cuda':
        torch.cuda.empty_cache()
print(f'DONE {len(results)} runs → {a.out}', flush=True)
