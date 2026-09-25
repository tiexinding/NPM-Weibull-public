"""Paper#5 App B.3 robustness of the q/k RoPE-pair read-out (zero GPU).
Compares, on the controlled-grid endpoints (tclock D1-D4 x seeds 1-3, step 5000, 6 layers), two pair-level
q/k profiles: (i) log-mean = mean of the two rows' log RMS (rows d and d+hd/2), the read-out of Fig 4(c) and
the path gain/balance ratio; (ii) pooled = log of the pair's pooled RMS sqrt((ms_d+ms_{d+hd/2})/2), which is
invariant under the common rotation of a rotary pair. Both are median-centred over the whole matrix
(as in the p5v2_identity_transfer_v1 pipeline), then the median over heads is taken.
Output: 08_paper5_draft/data/p5v3_pair_readout_robustness_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_paper5_draft/data/p5v3_pair_readout_robustness_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
H = 8

def layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True); st = st.get("model", st)
    L = {}
    for n, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.([qk]_proj)\.weight$", n)
        if m: L.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return L

rows = []
for a in ["D1", "D2", "D3", "D4"]:
    for s in [1, 2, 3]:
        for Lk, m in sorted(layers(ROOT / f"04_tclock_runs/wt_ckpts/tclock_{a}_s{s}/step5000.pt").items()):
            P = {}
            for nm in ("q_proj", "k_proj"):
                W = m[nm]; hd = W.shape[0] // H; half = hd // 2
                ms = (W ** 2).mean(1).reshape(H, hd); lr = 0.5 * np.log(ms)
                a1 = (lr[:, :half] + lr[:, half:]) / 2
                a2 = 0.5 * np.log((ms[:, :half] + ms[:, half:]) / 2)
                for key, x in (("logmean", a1), ("pooled", a2)):
                    P[nm, key] = np.median(x - np.median(x), 0)
            e = {"arm": a, "seed": s, "layer": Lk}
            for key in ("logmean", "pooled"):
                x, y = P["q_proj", key], P["k_proj", key]
                e[key + "_r"] = float(np.corrcoef(x, y)[0, 1])
                e[key + "_gain_balance"] = float((x + y).var() / (x - y).var())
            rows.append(e)
summ = {}
for key in ("logmean_r", "pooled_r", "logmean_gain_balance", "pooled_gain_balance"):
    v = np.array([r[key] for r in rows]); summ[key] = {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75))}
d = np.array([r["pooled_r"] - r["logmean_r"] for r in rows])
summ["pooled_minus_logmean_r"] = {"median": float(np.median(d)), "max_abs": float(np.abs(d).max())}
json.dump({"schema": "p5v3-pair-readout-robustness-v1", "script": Path(__file__).name, "script_sha256_16": SHA,
           "protocol": "tclock D1-D4 x seeds 1-3, step 5000, 6 layers (72 cells); pair profiles median-centred over the matrix, median over 8 heads; Pearson r of q vs k; gain/balance = var(g)/var(b) with g=x+y, b=x-y",
           "summary": summ, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summ, indent=1))
