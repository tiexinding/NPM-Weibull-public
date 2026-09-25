#!/usr/bin/env python3
"""P5 field-level counterfactual edit (zero training): does the paired path gain carry model function?

Model: ivn_base (LLaMA-style 70M, Gaussian init, seed 1, 30k steps), full_model_final.pt.
Paths (identity-axis channel shared by two adjacent projections):
  VO: v rows <-> o columns (head/value channel, 512 per layer)
  UD: up rows <-> down columns (FFN hidden unit, 1376 per layer)
  QK: q rows <-> k rows, taken at RoPE-pair level (rows d and d+32 of a head share one scale; 256 pairs per layer)
Fields: h = ln RMS along the identity axis, median-centred per matrix. gain g = h1 + h2, balance b = h1 - h2.
Edits (applied to all six layers of one path; each matrix then rescaled to its original Frobenius norm):
  balance  : side1 *= exp(-b/2), side2 *= exp(+b/2)  -> function-preserving identity (self-check, prediction: dloss = 0)
  flatten a: side1 *= exp(-a g/2), side2 *= exp(-a g/2), a in {0.25,0.5,1,1.5,2}; a=1 flat gain, a=2 reversed gain
  shuffle  : g -> g[pi], i.e. both sides *= exp((g[pi]-g)/2); same histogram / width, different arrangement (5 seeds)
  decile d : flatten (a=1) only the channels whose g lies in decile d (10 conditions)
Pre-registered predictions (written before the first run):
  P1 balance dloss ~ 0 (|d| < 1e-4).  P2 flatten loss rises monotonically in a.
  P3 shuffle vs flatten(a=1): if shuffle >= flatten, arrangement carries function beyond width.
  P4 top-gain decile costs more than bottom-gain decile.
Eval: 48 fixed sequences x 512 tokens from ea_tokens offset 90,000,000 (training sampled the whole array
with replacement, so this is in-distribution, not held-out); dloss = mean loss(edited) - mean loss(base).
"""
import json, sys, time, copy
from pathlib import Path
import numpy as np, torch
from transformers import LlamaConfig, LlamaForCausalLM
HERE = Path(__file__).resolve().parent
# Run from a local (ext4) copy of this file next to: full_model_final.pt (05_ivn_runs/ivn_base), ea_tokens.npy (data_local),
# llama70m_config.json (cloud_bundle). Output copied to 08_paper5_draft/data/p5v3_gain_edit_eval_v1.{json,log}.
torch.set_num_threads(20)
cfg = LlamaConfig(**json.load(open(HERE / "llama70m_config.json")))
SD = torch.load(HERE / "full_model_final.pt", map_location="cpu")
toks = np.load(HERE / "ea_tokens.npy", mmap_mode="r")
NSEQ, SEQ, BS, OFF = 48, 512, 8, 60_000_000
X = torch.tensor(np.array(toks[OFF:OFF + NSEQ * SEQ]).reshape(NSEQ, SEQ), dtype=torch.long)
L = cfg.num_hidden_layers; H = cfg.num_attention_heads; D = cfg.hidden_size // H

def model_from(sd):
    m = LlamaForCausalLM(cfg); m.load_state_dict(sd, strict=True); m.eval(); return m

@torch.no_grad()
def eval_loss(sd):
    m = model_from(sd); tot = 0.0
    for b in range(0, NSEQ, BS):
        x = X[b:b + BS]; tot += float(m(input_ids=x, labels=x).loss) * len(x)
    return tot / NSEQ

def fields(sd, layer, path):
    """return (W1 key, W2 key, axis1, axis2, h1, h2) with h median-centred ln RMS on the identity axis."""
    p = f"model.layers.{layer}."
    if path == "VO":
        k1, k2 = p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight"; ax1, ax2 = "row", "col"
    elif path == "UD":
        k1, k2 = p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"; ax1, ax2 = "row", "col"
    else:
        k1, k2 = p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight"; ax1, ax2 = "pair", "pair"
    def lnrms(W, ax):
        if ax == "row": r = W.pow(2).mean(1).sqrt()
        elif ax == "col": r = W.pow(2).mean(0).sqrt()
        else:  # RoPE pair: rows (h*D+d, h*D+d+D/2)
            r2 = W.pow(2).mean(1); r2 = r2.view(H, D); r = ((r2[:, :D // 2] + r2[:, D // 2:]) / 2).sqrt().reshape(-1)
        h = r.log(); return h - h.median()
    return k1, k2, ax1, ax2, lnrms(sd[k1], ax1), lnrms(sd[k2], ax2)

def scale(W, ax, f):
    """multiply W along identity axis by factor vector f (len = channels)."""
    if ax == "row": return W * f[:, None]
    if ax == "col": return W * f[None, :]
    fp = f.view(H, D // 2); ff = torch.cat([fp, fp], 1).reshape(-1); return W * ff[:, None]

def edit(sd, path, mode, a=1.0, rng=None, decile=None):
    out = {k: v for k, v in sd.items()}
    for l in range(L):
        k1, k2, ax1, ax2, h1, h2 = fields(sd, l, path)
        g, b = h1 + h2, h1 - h2
        if mode == "balance":
            f1, f2 = torch.exp(-b / 2), torch.exp(b / 2)
        elif mode == "flatten":
            f1 = f2 = torch.exp(-a * g / 2)
        elif mode == "shuffle":
            pi = torch.tensor(rng.permutation(len(g))); f1 = f2 = torch.exp((g[pi] - g) / 2)
        elif mode == "decile":
            q = torch.quantile(g, torch.tensor([decile / 10, (decile + 1) / 10]))
            sel = (g >= q[0]) & (g <= q[1]) if decile == 9 else (g >= q[0]) & (g < q[1])
            f1 = f2 = torch.where(sel, torch.exp(-g / 2), torch.ones_like(g))
        W1, W2 = scale(sd[k1], ax1, f1), scale(sd[k2], ax2, f2)
        if mode != "balance":  # keep each matrix at its original Frobenius norm
            W1 = W1 * (sd[k1].norm() / W1.norm()); W2 = W2 * (sd[k2].norm() / W2.norm())
        out[k1], out[k2] = W1, W2
    return out

def width(sd, path):
    return float(np.median([float((fields(sd, l, path)[4] + fields(sd, l, path)[5]).std()) for l in range(L)]))

if __name__ == "__main__":
    t0 = time.time(); res = {"base_loss": eval_loss(SD), "nseq": NSEQ, "seq": SEQ, "offset": OFF, "conditions": []}
    print("base", round(res["base_loss"], 5), flush=True)
    for path in ("VO", "UD", "QK"):
        res.setdefault("gain_width", {})[path] = width(SD, path)
        conds = [("balance", {})] + [("flatten", {"a": a}) for a in (0.25, 0.5, 1.0, 1.5, 2.0)] + \
                [("shuffle", {"seed": s}) for s in range(5)] + [("decile", {"decile": d}) for d in range(10)]
        for mode, kw in conds:
            sd2 = edit(SD, path, mode, a=kw.get("a", 1.0), rng=np.random.default_rng(kw.get("seed", 0)), decile=kw.get("decile"))
            loss = eval_loss(sd2); row = {"path": path, "mode": mode, **kw, "loss": loss, "dloss": loss - res["base_loss"]}
            res["conditions"].append(row); print(f"{path} {mode:8s} {kw} dloss {row['dloss']:+.5f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(HERE / "gain_edit_eval_slice2.json", "w"), indent=1); print("wrote json")
