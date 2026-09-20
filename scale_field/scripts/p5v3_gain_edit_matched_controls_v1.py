#!/usr/bin/env python3
"""P5 gain-edit perturbation-matched controls (zero training). Same checkpoint/paths/fields as p5v3_gain_edit_eval_v1.py.
Conditions per path (both sides of the path multiplied by exp(delta/2) per channel, Frobenius norm restored):
  flatten1        delta = -g                      (reference; identical to v1 a=1)
  amplify a       delta = -a g, a in {-0.5,-1}    (a=-1: field doubled, same |delta| as flatten1 -> sign of the move at matched magnitude)
  shuffle_matched delta = (g[pi]-g)/sqrt(2)       (arrangement changed, per-channel RMS of delta matched to flatten1; 5 seeds)
  gauss           delta = -eps, eps~N(0,std(g))   (no structure, RMS matched to flatten1; 5 seeds)
  signflip        delta = -sigma*g, sigma=+-1     (per-channel magnitude identical to flatten1, direction random; 5 seeds)
Each condition scored on slice 1 (offset 9e7) and slice 2 (offset 6e7); relF2 = ||dW||^2/||W||^2 over the 12 edited matrices recorded.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from transformers import LlamaConfig, LlamaForCausalLM
HERE = Path(__file__).resolve().parent
torch.set_num_threads(12)
cfg = LlamaConfig(**json.load(open(HERE / "llama70m_config.json")))
SD = torch.load(HERE / "full_model_final.pt", map_location="cpu")
toks = np.load(HERE / "ea_tokens.npy", mmap_mode="r")
NSEQ, SEQ, BS = 48, 512, 8
SLICES = {"slice1": 90_000_000, "slice2": 60_000_000}
X = {k: torch.tensor(np.array(toks[o:o + NSEQ * SEQ]).reshape(NSEQ, SEQ), dtype=torch.long) for k, o in SLICES.items()}
L = cfg.num_hidden_layers; H = cfg.num_attention_heads; D = cfg.hidden_size // H
M = LlamaForCausalLM(cfg); M.eval()

@torch.no_grad()
def eval_loss(sd):
    M.load_state_dict(sd, strict=True); out = {}
    for k, x in X.items():
        tot = 0.0
        for b in range(0, NSEQ, BS):
            xb = x[b:b + BS]; tot += float(M(input_ids=xb, labels=xb).loss) * len(xb)
        out[k] = tot / NSEQ
    return out

def fields(sd, layer, path):
    p = f"model.layers.{layer}."
    if path == "VO": k1, k2, ax1, ax2 = p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight", "row", "col"
    elif path == "UD": k1, k2, ax1, ax2 = p + "mlp.up_proj.weight", p + "mlp.down_proj.weight", "row", "col"
    else: k1, k2, ax1, ax2 = p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", "pair", "pair"
    def lnrms(W, ax):
        if ax == "row": r = W.pow(2).mean(1).sqrt()
        elif ax == "col": r = W.pow(2).mean(0).sqrt()
        else:
            r2 = W.pow(2).mean(1).view(H, D); r = ((r2[:, :D // 2] + r2[:, D // 2:]) / 2).sqrt().reshape(-1)
        h = r.log(); return h - h.median()
    return k1, k2, ax1, ax2, lnrms(sd[k1], ax1), lnrms(sd[k2], ax2)

def scale(W, ax, f):
    if ax == "row": return W * f[:, None]
    if ax == "col": return W * f[None, :]
    fp = f.view(H, D // 2); return W * torch.cat([fp, fp], 1).reshape(-1)[:, None]

def delta_for(mode, g, a, rng):
    if mode == "flatten1": return -g
    if mode == "amplify": return -a * g
    if mode == "shuffle_matched": pi = torch.tensor(rng.permutation(len(g))); return (g[pi] - g) / np.sqrt(2)
    if mode == "gauss": return -torch.tensor(rng.normal(0, float(g.std()), len(g)), dtype=g.dtype)
    if mode == "signflip": return -torch.tensor(rng.choice([-1.0, 1.0], len(g)), dtype=g.dtype) * g
    raise ValueError(mode)

def edit(sd, path, mode, a=1.0, seed=0):
    rng = np.random.default_rng(seed); out = dict(sd); num = den = 0.0
    for l in range(L):
        k1, k2, ax1, ax2, h1, h2 = fields(sd, l, path); g = h1 + h2
        f = torch.exp(delta_for(mode, g, a, rng) / 2)
        W1, W2 = scale(sd[k1], ax1, f), scale(sd[k2], ax2, f)
        W1 = W1 * (sd[k1].norm() / W1.norm()); W2 = W2 * (sd[k2].norm() / W2.norm())
        out[k1], out[k2] = W1, W2
        for k, W in ((k1, W1), (k2, W2)): num += float((W - sd[k]).pow(2).sum()); den += float(sd[k].pow(2).sum())
    return out, num / den

if __name__ == "__main__":
    t0 = time.time(); base = eval_loss(SD); res = {"base_loss": base, "nseq": NSEQ, "seq": SEQ, "slices": SLICES, "conditions": []}
    print("base", base, flush=True)
    for path in ("VO", "UD", "QK"):
        conds = [("flatten1", {})] + [("amplify", {"a": a}) for a in (-0.5, -1.0)] + \
                [(m, {"seed": s}) for m in ("shuffle_matched", "gauss", "signflip") for s in range(5)]
        for mode, kw in conds:
            sd2, relF2 = edit(SD, path, mode, a=kw.get("a", 1.0), seed=kw.get("seed", 0))
            loss = eval_loss(sd2); row = {"path": path, "mode": mode, **kw, "relF2": relF2,
                                          "loss": loss, "dloss": {k: loss[k] - base[k] for k in loss}}
            res["conditions"].append(row)
            print(f"{path} {mode:15s} {kw} relF2 {relF2:.4f} dloss s1 {row['dloss']['slice1']:+.5f} s2 {row['dloss']['slice2']:+.5f} [{time.time()-t0:.0f}s]", flush=True)
            json.dump(res, open(HERE / "gain_edit_matched_controls_v1.json", "w"), indent=1)
    print("done", flush=True)
