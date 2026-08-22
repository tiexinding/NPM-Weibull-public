#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import json, os, glob, re, argparse
import numpy as np
from pathlib import Path

ROOT = Path(os.environ.get('NPM_ROOT', '.'))
P4 = ROOT / '50_data/05_paper4'

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=str(P4 / 'runs'))
ap.add_argument('--tok', default=str(P4 / 'token_data/p4grid'))
ap.add_argument('--metrics', default=str(P4 / 'gate_out/grid_trainlevel_metrics.json'))
ap.add_argument('--out', default=str(P4 / 'data/p4_grid_eval.json'))
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--blk', type=int, default=512)
a = ap.parse_args()

print('[1] importing torch...', flush=True)
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
torch.set_grad_enabled(False)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'    torch ok, device={dev}', flush=True)

cfg = GPTNeoXConfig(vocab_size=50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                    intermediate_size=2048, rotary_pct=0.25, rotary_emb_base=10000,
                    max_position_embeddings=2048, use_parallel_residual=True,
                    layer_norm_eps=1e-5, hidden_act='gelu')

#
gm = json.load(open(a.metrics))
DR = {r['name']: (r['D'], r['R']) for r in gm['rows']}

#
BLK = a.blk
def load_held(p):
    h = np.load(p).astype(np.int64)
    nb = len(h) // BLK
    return torch.tensor(h[:nb * BLK].reshape(nb, BLK), dtype=torch.long)
X_clean = load_held(f'{a.tok}/held_clean.npy')
X_match = {'000': X_clean}
for ft in ('025', '050', '100'):
    p = f'{a.tok}/held_f{ft}.npy'
    if os.path.exists(p):
        X_match[ft] = load_held(p)
print(f'[2] held-out: clean {X_clean.shape[0]}×{BLK} + matched f∈{sorted(X_match)}', flush=True)

def ce_loss(m, X):
    tot, ntok = 0.0, 0
    for i in range(X.shape[0]):
        ids = X[i:i + 1].to(dev)
        out = m(ids, labels=ids)
        tot += float(out.loss) * (BLK - 1); ntok += (BLK - 1)
    return tot / ntok

#
rows = []
if os.path.exists(a.out):
    rows = json.load(open(a.out))
    print(f'[resume] resumed with  {len(rows)} run', flush=True)
done = {r['name'] for r in rows}
os.makedirs(os.path.dirname(a.out), exist_ok=True)

for d in sorted(glob.glob(f'{a.runs}/p4grid_*/')):
    name = d.rstrip('/').split('/')[-1]
    if name in done:
        continue
    m = re.match(r'p4grid_f(\d{3})_r(\d{2})_s(\d+)', name)
    if not m:
        continue
    ft, rho, seed = m.group(1), int(m.group(2)), int(m.group(3))
    ck = f'{d}W_ckpts/step{a.steps}.pt'
    lt = f'{d}lambda_trajectory_continue.json'
    st = f'{d}v1b_spline_true.jsonl'
    if not all(os.path.exists(x) for x in (ck, lt, st)):
        print(f'  [skip incomplete] {name}', flush=True)
        continue
    corp = f'p4grid_f{ft}_r{rho:02d}_s0'
    D, R = DR[corp]
    lam = json.load(open(lt))
    los = [json.loads(l) for l in open(st)]
    l0, lf = lam[0]['lambda'], lam[-1]['lambda']
    tloss = los[-1]['loss']
    max_neg_align = max(-r['true_align_full'] for r in los)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model = GPTNeoXForCausalLM(cfg)
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.to(dev).eval()
    held = ce_loss(model, X_clean)
    held_m = ce_loss(model, X_match[ft]) if ft in X_match else float('nan')
    del model, sd
    r = dict(name=name, f=int(ft) / 100, rho=rho, seed=seed, D=D, R=R,
             dlam2=(lf ** 2 - l0 ** 2) * 1e4, lam0=l0, lamf=lf,
             train_loss=tloss, held_loss=held, held_matched=held_m,
             gap=held - tloss, gap_matched=held_m - tloss,
             max_neg_align=max_neg_align, unstable=bool(max_neg_align > 10),
             miss=len(miss), unexp=len(unexp))
    rows.append(r)
    print(f"  [{len(rows):2d}] {name}: held={held:.3f} matched={held_m:.3f} train={tloss:.3f} "
          f"gap={held - tloss:+.3f} Δλ²={r['dlam2']:+.2f} D={D:.2f} R={R:.3f} "
          f"{'🔴unstable' if r['unstable'] else ''}", flush=True)
    json.dump(rows, open(a.out, 'w'), ensure_ascii=False, indent=1)

#
ok = [r for r in rows if not r['unstable']]
if len(ok) >= 8:
    D = np.array([r['D'] for r in ok]); R = np.array([r['R'] for r in ok])
    DL = np.array([r['dlam2'] for r in ok]); HL = np.array([r['held_loss'] for r in ok])
    TL = np.array([r['train_loss'] for r in ok]); GG = np.array([r['gap'] for r in ok])
    GM = np.array([r['gap_matched'] for r in ok])
    def pc(x, y): return float(np.corrcoef(x, y)[0, 1])
    def partial(x, y, Z):
        Z = np.column_stack(list(Z) + [np.ones(len(x))])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    n_uns = len(rows) - len(ok)
    print(f'\n===== grid results (n={len(ok)}, unstable excluded {n_uns}) =====', flush=True)
    print(f'corr(Δλ², held)              = {pc(DL, HL):+.3f}', flush=True)
    print(f'pcorr(Δλ², held | D)         = {partial(DL, HL, [D]):+.3f}   (probe protocol, controlling D only)', flush=True)
    print(f'★ pcorr(Δλ², held | D, R)    = {partial(DL, HL, [D, R]):+.3f}   (dual control = the beyond-D criterion)', flush=True)
    print(f'★ pcorr(Δλ², gap  | D, R)    = {partial(DL, GG, [D, R]):+.3f}', flush=True)
    print(f'pcorr(Δλ², gap_matched | D,R)= {partial(DL, GM, [D, R]):+.3f}   (structure-matched gap, removes shift confound)', flush=True)
    print(f'corr(R, gap) = {pc(R, GG):+.3f}  | corr(D, held) = {pc(D, HL):+.3f}', flush=True)
    #
    print('\n  (f, ρ) → Δλ² / gap  [joint lambda+gap classification]:', flush=True)
    for r in sorted(ok, key=lambda r: (r['f'], r['rho'])):
        print(f"   f={r['f']:.2f} ρ={r['rho']:>2}: Δλ²={r['dlam2']:+7.2f} gap={r['gap']:+6.2f} "
              f"held={r['held_loss']:6.2f}", flush=True)
else:
    print(f'\n[analysis skipped] insufficient runs ({len(ok)})', flush=True)
print('DONE', flush=True)
