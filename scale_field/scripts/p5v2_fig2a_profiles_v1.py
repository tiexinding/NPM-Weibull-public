"""Paper#5 v2 Figure 2(a) input: the q/k/v frequency profiles of the IVN base and ropeperm arms.

Profile definition = the pre-registered J3 profile of scripts/ivn_j1_j3_hw_v1.py, reproduced here
verbatim: cumulative median-centred Delta log(row RMS) over steps_J1 (3200 -> 30000), averaged over
the two rows of each rotary pair (j, j+32), median over layers x heads (pooled) or over heads
(per layer). Coordinate index m carries frequency pi[m] in the ropeperm arm
(inv_freq_after[m] = inv_freq_before[pi[m]]), so the frequency-indexed profile is P_coord[pi_inv].
The recomputed C_freq / C_coord are asserted against 05_ivn_runs/ivn_j1_j3_hw_v1.json.

Output: 08_paper5_draft/data/p5v2_fig2a_profiles_v1.json
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
J = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
OUT = ROOT / "08_paper5_draft/data/p5v2_fig2a_profiles_v1.json"
H, DH, NF = J["heads"], J["head_dim"], J["n_freq_pairs"]
assert DH == 2 * NF
STEPS = J["steps_J1"]
KINDS = ["q_proj", "k_proj", "v_proj"]
ARMS = ["base", "ropeperm"]
pi = np.array(J["pi"], dtype=int)
assert sorted(pi.tolist()) == list(range(NF))
pi_inv = np.empty(NF, dtype=int); pi_inv[pi] = np.arange(NF)


def cum_rel_drift(arm):
    logr = []
    for s in STEPS:
        st = torch.load(ROOT / f"05_ivn_runs/ivn_{arm}/ckpt/step{s}.pt", map_location="cpu", weights_only=True)
        cur = {}
        for nm, tw in st.items():
            proj = nm.rsplit(".", 2)[-2]
            if proj not in KINDS:
                continue
            layer = int(nm.split("layers.")[1].split(".")[0])
            w = tw.to(torch.float64)
            cur[(proj, layer)] = np.log(np.sqrt((w ** 2).mean(dim=1).numpy()))
        assert len(cur) == 3 * 6, (arm, s, len(cur))
        logr.append(cur)
    X = {}
    for key in logr[0]:
        d = np.zeros_like(logr[0][key])
        for a, b in zip(logr[:-1], logr[1:]):
            dd = b[key] - a[key]
            d += dd - np.median(dd)
        X[key] = d
    return X


def pair_avg(x):
    out = np.empty(H * NF)
    for h in range(H):
        blk = x[h * DH:(h + 1) * DH]
        out[h * NF:(h + 1) * NF] = (blk[:NF] + blk[NF:]) / 2.0
    return out


def profiles(X, kind):
    per_layer, allv = {}, {m: [] for m in range(NF)}
    for (kd, layer), x in X.items():
        if kd != kind:
            continue
        xp = pair_avg(x)
        per_layer[layer] = [float(np.median([xp[h * NF + m] for h in range(H)])) for m in range(NF)]
        for idx in range(len(xp)):
            allv[idx % NF].append(xp[idx])
    pooled = [float(np.median(allv[m])) for m in range(NF)]
    return pooled, {str(L): per_layer[L] for L in sorted(per_layer)}


X = {arm: cum_rel_drift(arm) for arm in ARMS}
res = {"schema": "p5v2-fig2a-profiles-v1",
       "definition": ("J3 profile of scripts/ivn_j1_j3_hw_v1.py: cumulative median-centred Delta log(row RMS) over "
                      "steps_J1, rotary-pair average of rows (j, j+NF), median over layers x heads (pooled) / heads (per layer); "
                      "coordinate index m of the ropeperm arm carries frequency pi[m]; frequency-indexed = coord[pi_inv]"),
       "steps_J1": STEPS, "pi": pi.tolist(), "pi_inv": pi_inv.tolist(), "pi_direction": J["pi_direction"],
       "heads": H, "head_dim": DH, "n_freq_pairs": NF, "arms": ARMS, "kinds": KINDS,
       "pooled": {}, "per_layer": {}, "similarity_recomputed": {}, "similarity_prereg_json": {}}
for kd in KINDS:
    Pb, Lb = profiles(X["base"], kd)
    Pc, Lc = profiles(X["ropeperm"], kd)
    Pf = [Pc[i] for i in pi_inv]
    c_f = float(np.corrcoef(Pf, Pb)[0, 1]); c_c = float(np.corrcoef(Pc, Pb)[0, 1])
    ref = J["J3"][kd]
    assert abs(c_f - ref["C_freq"]) < 1e-3 and abs(c_c - ref["C_coord"]) < 1e-3, (kd, c_f, ref["C_freq"], c_c, ref["C_coord"])
    res["pooled"][kd] = {"base": Pb, "ropeperm_by_coordinate": Pc, "ropeperm_by_frequency": Pf}
    res["per_layer"][kd] = {"base": Lb, "ropeperm_by_coordinate": Lc,
                            "ropeperm_by_frequency": {L: [Lc[L][i] for i in pi_inv] for L in Lc}}
    res["similarity_recomputed"][kd] = {"C_freq": c_f, "C_coord": c_c, "dC": c_f - c_c}
    res["similarity_prereg_json"][kd] = {"C_freq": ref["C_freq"], "C_coord": ref["C_coord"], "null_dC_p95": ref["null_dC_p95"]}
res["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
for kd in KINDS:
    print(kd, {k: round(v, 4) for k, v in res["similarity_recomputed"][kd].items()},
          "range base", round(min(res["pooled"][kd]["base"]), 3), round(max(res["pooled"][kd]["base"]), 3))
