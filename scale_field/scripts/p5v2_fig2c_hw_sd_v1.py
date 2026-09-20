"""Paper#5 v2 (260911): Figure 2(c) row-scale spread in the sd definition.

The main text uses one heterogeneity statistic, H^W = sd_i(ln r_i) with r_i the RMS of row i
(the definition of kmix_bridge_v1_1 / Figure 1(a) / Figure 3). Figure 2(c) previously showed the
pre-registered IVN statistic H^W_IQR = IQR_i(log10 r_i^2); that panel moves to the appendix (S5).

Here H^W (sd), k_raw and k_norm are recomputed from the IVN base / ropeperm checkpoints over the
terminal window {20k, 25k, 30k}, per matrix, and cross-checked against
08_paper5_draft/data/p5_kmix_on_ivn_v1.json (same protocol, computed 260901).
Aggregation follows ivn_j1_j3_hw_v1.py: per family, median over layers at each step, then median
over the terminal window.  o_proj has no row-side read-out (column-axis identity).

Outputs: 08_paper5_draft/data/p5v2_fig2c_hw_sd_readouts_v1.json
"""
import json, re, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_paper5_draft/data/p5v2_fig2c_hw_sd_readouts_v1.json"
J = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
TERM = J["steps_terminal"]
KM = json.load(open(ROOT / "08_paper5_draft/data/p5_kmix_on_ivn_v1.json"))
KROWS = {(r["arm"], r["step"], r["mat"]): r for r in KM["per_matrix_readouts"]}
HB, HR = 1024, (-12.0, 2.0)
FAM = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
ARMS = ("base", "ropeperm")


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


def hw_sd(w):
    return float(np.log(np.sqrt((w ** 2).mean(1))).std())


def hw_iqr(w):
    v = np.log10((w ** 2).mean(1)); return float(np.percentile(v, 75) - np.percentile(v, 25))


rows = []
for arm in ARMS:
    for s in TERM:
        st = torch.load(ROOT / f"05_ivn_runs/ivn_{arm}/ckpt/step{s}.pt", map_location="cpu", weights_only=True)
        for nm, w in st.items():
            mm = re.search(r"\.(\w+_proj)\.weight$", nm)
            if not mm or mm.group(1) not in FAM:
                continue
            w = w.double().numpy()
            kr, r2r = kfit(w); kn, r2n = kfit(rownorm(w))
            L = int(re.search(r"layers\.(\d+)", nm).group(1))
            row = {"arm": arm, "step": s, "L": L, "kind": mm.group(1), "mat": nm, "k_raw": kr, "k_norm": kn,
                   "r2fit_raw": r2r, "r2fit_norm": r2n, "hw_sd": hw_sd(w), "hw_iqr_log10_rms2": hw_iqr(w)}
            ref = KROWS[(arm, s, nm)]
            assert abs(ref["hw"] - row["hw_sd"]) < 1e-6 and abs(ref["k_raw"] - kr) < 1e-6 \
                and abs(ref["k_norm"] - kn) < 1e-6, ("cross-check vs p5_kmix_on_ivn_v1", arm, s, nm)
            rows.append(row)
assert len(rows) == len(ARMS) * len(TERM) * 6 * len(FAM), len(rows)
assert min(min(r["r2fit_raw"], r["r2fit_norm"]) for r in rows) > 0.99, "fit gate"


def agg(arm, kind, key, steps):
    per_step = [np.median([r[key] for r in rows if r["arm"] == arm and r["kind"] == kind and r["step"] == s])
                for s in steps]
    assert len(per_step) == len(steps)
    return float(np.median(per_step))


fam = {}
for kd in FAM:
    d = {}
    for key in ("hw_sd", "hw_iqr_log10_rms2", "k_raw", "k_norm"):
        b, p = agg("base", kd, key, TERM), agg("ropeperm", kd, key, TERM)
        d[key] = {"base": b, "ropeperm": p, "pct_change": 100 * (p / b - 1)}
    # pooled aggregation (median over the 6 layers x 3 terminal steps = 18 matrices): the aggregation used
    # for the k / k_norm bars of Figure 2(c) (p5v2_fig2_v1.py); the panel uses this one for all three bars
    d["pooled18"] = {}
    for key in ("hw_sd", "hw_iqr_log10_rms2", "k_raw", "k_norm"):
        vb = [r[key] for r in rows if r["arm"] == "base" and r["kind"] == kd]
        vp = [r[key] for r in rows if r["arm"] == "ropeperm" and r["kind"] == kd]
        assert len(vb) == len(vp) == 6 * len(TERM)
        b, p = float(np.median(vb)), float(np.median(vp))
        d["pooled18"][key] = {"base": b, "ropeperm": p, "pct_change": 100 * (p / b - 1)}
    b30, p30 = agg("base", kd, "hw_sd", [TERM[-1]]), agg("ropeperm", kd, "hw_sd", [TERM[-1]])
    d["hw_sd_step30000_only"] = {"base": b30, "ropeperm": p30, "pct_change": 100 * (p30 / b30 - 1)}
    # the IQR recomputed here must match the stored pre-registered IVN value
    for arm in ARMS:
        assert abs(d["hw_iqr_log10_rms2"][arm] - J["H_W"][arm][kd]) < 1e-6, ("IQR cross-check", arm, kd)
    fam[kd] = d
res = {"schema": "p5v2-fig2c-hw-sd-v1",
       "definition": {"hw_sd": "sd over rows of ln(row RMS); identical to kmix_bridge_v1_1 'hw' (Figure 1a, Figure 3)",
                      "hw_iqr_log10_rms2": "IQR over rows of log10(row RMS^2); pre-registered IVN J2 statistic (ivn_j1_j3_hw_v1)",
                      "aggregation": "per family: median over 6 layers at each terminal step, then median over the terminal window (keys hw_sd/...); 'pooled18' = median over the 18 terminal matrices, the aggregation used for every bar of Figure 2(c)",
                      "terminal_window": TERM, "fit": "weibull_fit_1024 middle-80%, R^2 gate 0.99 (all pass)"},
       "per_family": fam, "per_matrix": rows,
       "stored_HW_IQR_for_reference": {a: J["H_W"][a] for a in ARMS},
       "provenance": {"script": Path(__file__).name,
                      "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                      "sources": ["05_ivn_runs/ivn_{base,ropeperm}/ckpt/step{20000,25000,30000}.pt",
                                  "05_ivn_runs/ivn_j1_j3_hw_v1.json", "08_paper5_draft/data/p5_kmix_on_ivn_v1.json"]}}
json.dump(res, open(OUT, "w"), indent=1)
print("->", OUT)
for kd in FAM:
    d = fam[kd]
    print(f"{kd:10s} hw_sd {d['hw_sd']['base']:.4f}->{d['hw_sd']['ropeperm']:.4f} ({d['hw_sd']['pct_change']:+.1f}%) "
          f"| 30k only {d['hw_sd_step30000_only']['pct_change']:+.1f}% | IQR {d['hw_iqr_log10_rms2']['pct_change']:+.1f}% "
          f"| k {d['k_raw']['pct_change']:+.2f}% k_norm {d['k_norm']['pct_change']:+.2f}% "
          f"|| pooled18: hw_sd {d['pooled18']['hw_sd']['pct_change']:+.1f}% k {d['pooled18']['k_raw']['pct_change']:+.2f}% k_norm {d['pooled18']['k_norm']['pct_change']:+.2f}%")
