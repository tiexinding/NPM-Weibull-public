"""Public Pythia endpoints (70m / 160m / 410m / 1b): identity-axis-normalized refit (260910, zero GPU).

Closes the K3 open item ("public checkpoints carry no k_norm") and adds 70m, which phase1b
skipped only because its step-0 safetensors was missing (the endpoint-only refit needs no step 0).

Per block and kind (q, k, v split from the fused per-head QKV as in phase1b; o; ffn_in; ffn_out):
  k_raw      middle-80% Weibull shape of |W| (weibull_fit_1024, same protocol as Figure 1a/b)
  k_axis     same fit after normalizing along the identity axis: rows for q/k/v/ffn_in,
             columns for o/ffn_out (each row/column to the matrix RMS; global RMS preserved)
  k_both     after alternating row/column normalization to convergence (0.1%)
  H_row, H_col = sd of log row / column RMS
Cross-check: k_raw is compared with phase1b's `kf` (different fitter) and the difference recorded.

Sources: ~/.cache/huggingface/hub/models--EleutherAI--pythia-{70m,160m,410m,1b} (final revision;
         160m = step143000 as in phase1b), 03_existing_data/phase1b_multisize_endpoints_v1.json
Output:  08_paper5_draft/data/p5_public_pythia_knorm_v1.json
Grade:   static public endpoints, no seeds, no trajectory (K3).
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_paper5_draft/data/p5_public_pythia_knorm_v1.json"
HUB = Path.home() / ".cache/huggingface/hub"
HB, HR = 1024, (-12.0, 2.0)
SIZES = {
    "70m": dict(hidden=512, heads=8, layers=6, ref="main"),
    "160m": dict(hidden=768, heads=12, layers=12, ref="step143000"),
    "410m": dict(hidden=1024, heads=16, layers=24, ref="main"),
    "1b": dict(hidden=2048, heads=8, layers=16, ref="main"),
}
ROW_KINDS = ("q", "k", "v", "ffn_in")
COL_KINDS = ("o", "ffn_out")


def fit_hist(hist):
    hist = hist.astype(np.float64); total = hist.sum()
    edges = np.linspace(HR[0], HR[1], HB + 1)
    F = ((np.cumsum(hist) / total) + ((np.cumsum(hist) - hist) / total)) / 2.0
    bc = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    m = (F >= 0.1) & (F <= 0.9) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc[m], np.ones(m.sum())]).T
    w = hist[m]; WX = X * w[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    res = y - X @ beta; ybar = (w * y).sum() / w.sum()
    return float(beta[0]), float(1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum())


def kfit(x):
    return fit_hist(np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0])


def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True)); z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())


def colnorm(w):
    return rownorm(w.T).T


def bothnorm(w, it=60, tol=1e-5):
    scale = np.sqrt((w ** 2).mean())
    for _ in range(it):
        w2 = colnorm(rownorm(w))
        done = np.abs(w2 - w).max() < tol * scale
        w = w2
        if done:
            break
    rr = np.sqrt((w ** 2).mean(1)); cc = np.sqrt((w ** 2).mean(0))
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, "not converged"
    return w


def st_path(model, ref):
    base = HUB / f"models--EleutherAI--{model}"
    snap = (base / "refs" / ref).read_text().strip()
    p = glob.glob(str(base / "snapshots" / snap / "*.safetensors"))
    assert p, (model, ref, snap)
    return p[0]


def mats(path, hidden, heads, layers):
    hd = hidden // heads
    with safe_open(path, framework="np") as sf:
        for b in range(layers):
            pre = f"gpt_neox.layers.{b}"
            qkv = sf.get_tensor(f"{pre}.attention.query_key_value.weight").astype(np.float64)
            assert qkv.shape == (3 * hidden, hidden)
            w3 = qkv.reshape(heads, 3, hd, hidden)  # per-head [q,k,v] blocks, as in phase1b / HF
            yield b, "q", w3[:, 0].reshape(heads * hd, hidden)
            yield b, "k", w3[:, 1].reshape(heads * hd, hidden)
            yield b, "v", w3[:, 2].reshape(heads * hd, hidden)
            yield b, "o", sf.get_tensor(f"{pre}.attention.dense.weight").astype(np.float64)
            yield b, "ffn_in", sf.get_tensor(f"{pre}.mlp.dense_h_to_4h.weight").astype(np.float64)
            yield b, "ffn_out", sf.get_tensor(f"{pre}.mlp.dense_4h_to_h.weight").astype(np.float64)


P1B = {(r["size"], r["block"], r["kind"]): r["kf"]
       for r in json.load(open(ROOT / "03_existing_data/phase1b_multisize_endpoints_v1.json"))["rows"]}

rows = []
t0 = time.time()
for size, cfg in SIZES.items():
    path = st_path(f"pythia-{size}", cfg["ref"])
    n = 0
    for b, kind, w in mats(path, cfg["hidden"], cfg["heads"], cfg["layers"]):
        kr, r2r = kfit(w)
        wa = rownorm(w) if kind in ROW_KINDS else colnorm(w)
        ka, r2a = kfit(wa)
        kb, r2b = kfit(bothnorm(w))
        assert abs(np.sqrt((wa ** 2).mean()) - np.sqrt((w ** 2).mean())) < 1e-9
        rows.append({"size": size, "block": b, "kind": kind, "shape": list(w.shape),
                     "identity_axis": "row" if kind in ROW_KINDS else "col",
                     "k_raw": kr, "k_axis": ka, "k_both": kb,
                     "r2_raw": r2r, "r2_axis": r2a, "r2_both": r2b,
                     "H_row": float(np.log(np.sqrt((w ** 2).mean(1))).std()),
                     "H_col": float(np.log(np.sqrt((w ** 2).mean(0))).std()),
                     "phase1b_kf": P1B.get((size, b, kind))})
        n += 1
    assert n == cfg["layers"] * 6, (size, n)
    print(size, "done", n, "matrices", round(time.time() - t0), "s", flush=True)

assert len(rows) == 6 * sum(c["layers"] for c in SIZES.values())
gate = [r for r in rows if min(r["r2_raw"], r["r2_axis"], r["r2_both"]) < 0.99]
cmp_ = [abs(r["k_raw"] - r["phase1b_kf"]) for r in rows if r["phase1b_kf"] is not None]
summary = {}
for size in SIZES:
    summary[size] = {}
    for kind in ROW_KINDS + COL_KINDS:
        s = [r for r in rows if r["size"] == size and r["kind"] == kind]
        summary[size][kind] = {k: [float(np.median([r[k] for r in s])), float(min(r[k] for r in s)), float(max(r[k] for r in s))]
                               for k in ("k_raw", "k_axis", "k_both", "H_row", "H_col")}
json.dump({"schema": "p5-public-pythia-knorm-v1", "protocol": "weibull_fit_1024 middle-80%; identity-axis normalization rows q/k/v/ffn_in, cols o/ffn_out; both = alternating to 0.1%",
           "sizes": SIZES, "n_rows": len(rows), "fit_gate_below_0.99": [(r["size"], r["block"], r["kind"]) for r in gate],
           "phase1b_kf_absdiff_median_max": [float(np.median(cmp_)), float(max(cmp_))] if cmp_ else None,
           "summary_median_min_max": summary, "rows": rows}, open(OUT, "w"), indent=1)
print("gate fails", len(gate), "| |k_raw - phase1b kf| median/max", round(float(np.median(cmp_)), 4), round(float(max(cmp_)), 4))
for size in SIZES:
    print(size, {k: "raw %.3f axis %.3f both %.3f" % tuple(summary[size][k][x][0] for x in ("k_raw", "k_axis", "k_both")) for k in ("q", "k", "v", "o", "ffn_in", "ffn_out")})
