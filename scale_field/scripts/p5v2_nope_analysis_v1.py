"""NoPE boundary experiment (260912 cloud batch): llama-70M trained with rotary disabled (cos=1, sin=0),
otherwise identical to the IVN base run (same init, data stream, optimizer), compared with ivn_base.

Questions (descriptive, pre-set):
  (1) do q/k still develop a row-scale field of the same size as with RoPE?  -> H_row(nope)/H_row(base)
  (2) is there any frequency-like arrangement left?  the (j, j+32) "pseudo-pair" profile of the NoPE run is
      only a coordinate label; its correlation with the base run's true RoPE-pair profile is compared with
      the base cross-seed reference (S1: q .973, k .977) and a 32-point permutation null.
      head-level structure: share of row-profile variance carried by the 8 head means.
      q<->k row-level and pair-level correlation in both arms.
  (3) S10 unit transfer (gate/up rows <-> down columns, v rows <-> o columns) in both arms.
  (4) S13 path gain/balance var(g)/var(b) with re-pairing null in both arms.
  (5) loss curves.

Protocol functions are exec'd verbatim from the reference scripts (no re-implementation):
  p5v2_twoaxis_bridge_v1.py   fit_hist/kfit/rownorm/colnorm/bothnorm_tracked/analyse  (HB, HR)
  p5v2_path_gain_balance_v1.py prof/pair_profile/head_profile/gb_stats/null_ratio      (HEADS, HEAD_DIM)
Sources (read-only):
  10_p5batch_cloud/results/nope_s1/ckpt/step*.pt, loss.jsonl, run_meta.json
  05_ivn_runs/ivn_base/ckpt/step*.pt, loss.jsonl ; 05_ivn_runs/ivn_j1_j3_hw_v1.json (heads, head_dim)
  08_paper5_draft/data/p5v2_supp_readouts_v1.json  (S1 cross-run pair-profile reference)
Outputs:
  10_p5batch_cloud/results/nope_analysis_v1.json / .png
"""
import hashlib
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
NOPE = ROOT / "10_p5batch_cloud/results/nope_s1"
BASE = ROOT / "05_ivn_runs/ivn_base"
OUT_JSON = ROOT / "10_p5batch_cloud/results/nope_analysis_v1.json"
OUT_PNG = ROOT / "10_p5batch_cloud/results/nope_analysis_v1.png"
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LABEL = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
IDENT_AXIS = {"q_proj": 1, "k_proj": 1, "v_proj": 1, "gate_proj": 1, "up_proj": 1, "o_proj": 0, "down_proj": 0}  # 1 = rows
C_BASE, C_NOPE = "#404040", "#D95F02"
C_SEL, C_TRA = "#B2182B", "#2166AC"

# ---------------- protocol functions, exec'd verbatim from the reference scripts ----------------
def _exec_segment(path, start, stop, ns):
    src = (ROOT / "scripts" / path).read_text()
    i = src.index(start); j = src.index(stop, i)
    seg = src[i:j]; ns["__seg_sha16__" + path] = hashlib.sha256(seg.encode()).hexdigest()[:16]
    exec(compile(seg, str(path), "exec"), ns)

IVN = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
NS = {"np": np, "HB": 1024, "HR": (-12.0, 2.0), "HEADS": int(IVN["heads"]), "HEAD_DIM": int(IVN["head_dim"]),
      "N_NULL": 200, "RNG": np.random.default_rng(0)}
_src_tw = (ROOT / "scripts/p5v2_twoaxis_bridge_v1.py").read_text()
_hb, _hr = re.search(r"^HB, HR = (\d+), \(([^)]*)\)", _src_tw, re.M).groups()
assert int(_hb) == NS["HB"] and tuple(float(x) for x in _hr.split(",")) == NS["HR"], "fit-protocol constants drifted"
_exec_segment("p5v2_twoaxis_bridge_v1.py", "def fit_hist(", "def do_ckpt(", NS)
_exec_segment("p5v2_path_gain_balance_v1.py", "def prof(", "def profiles(", NS)
analyse, prof, pair_profile, head_profile, gb_stats, null_ratio = (NS[k] for k in ("analyse", "prof", "pair_profile", "head_profile", "gb_stats", "null_ratio"))
HEADS, HEAD_DIM = NS["HEADS"], NS["HEAD_DIM"]
RNG = np.random.default_rng(1)

S1 = json.load(open(ROOT / "08_paper5_draft/data/p5v2_supp_readouts_v1.json"))["S1"]
REF_CROSS = {"q_proj": float(S1["q"]["cross_run_pearson_median"]), "k_proj": float(S1["k"]["cross_run_pearson_median"])}


def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    assert len(lay) == 6 and all(set(v) == set(KINDS) for v in lay.values()), (p, sorted(lay))
    return lay


def steps_of(d):
    return sorted(int(re.search(r"step(\d+)\.pt", f.name).group(1)) for f in (d / "ckpt").glob("step*.pt"))


def loss_curve(d):
    rows = [json.loads(l) for l in open(d / "loss.jsonl") if l.strip()]
    return [(int(r["step"]), float(r["loss"])) for r in rows]


# ---------------- (1) two-sided fits per matrix ----------------
def _fit_job(args):
    arm, step, L, kind, w = args
    e = analyse(w); e.update({"arm": arm, "step": step, "layer": L, "kind": kind}); return e


def main():
    meta_n = json.load(open(NOPE / "run_meta.json")); assert meta_n["arm"] == "nope", meta_n["arm"]
    st_n, st_b = steps_of(NOPE), steps_of(BASE)
    common = sorted(set(st_n) & set(st_b))
    assert set([0, 800, 3200, 5000, 10000, 30000]) <= set(common), (st_n, st_b)
    common = [s for s in common if s in (0, 800, 3200, 5000, 10000, 30000)]
    lay = {("nope", s): load_layers(NOPE / f"ckpt/step{s}.pt") for s in common}
    lay.update({("base", s): load_layers(BASE / f"ckpt/step{s}.pt") for s in common})

    jobs = [(arm, s, L, k, lay[(arm, s)][L][k]) for (arm, s) in lay for L in range(6) for k in KINDS]
    assert len(jobs) == 2 * len(common) * 42, len(jobs)
    with Pool(8) as pool:
        fits = pool.map(_fit_job, jobs, chunksize=8)
    for e in fits:
        assert min(e["r2_raw"], e["r2_row"], e["r2_col"], e["r2_bi"]) > 0.98, (e["arm"], e["step"], e["layer"], e["kind"], e["r2_raw"])

    def med(arm, step, kind, key):
        return float(np.median([e[key] for e in fits if e["arm"] == arm and e["step"] == step and e["kind"] == kind]))
    H_ident = {arm: {k: {str(s): med(arm, s, k, "H_row" if IDENT_AXIS[k] else "H_col") for s in common} for k in KINDS} for arm in ("base", "nope")}
    K_tab = {arm: {k: {str(s): {kk: med(arm, s, k, kk) for kk in ("k_raw", "k_row", "k_col", "k_bi", "H_row", "H_col")} for s in common} for k in KINDS} for arm in ("base", "nope")}

    # ---------------- (2) q/k structure ----------------
    def qk_struct(arm, s):
        out = []
        for L in range(6):
            m = lay[(arm, s)][L]; m0 = lay[(arm, 0)][L]
            q_r, k_r = prof(m["q_proj"], 1), prof(m["k_proj"], 1)
            q_c, k_c = prof(m["q_proj"], 1) - prof(m0["q_proj"], 1), prof(m["k_proj"], 1) - prof(m0["k_proj"], 1)
            e = {"layer": L}
            for nm, h in (("q", q_r), ("k", k_r)):
                hp = pair_profile(h); hh = head_profile(h)
                e[f"{nm}_pair_static"] = hp.tolist(); e[f"{nm}_head_static"] = hh.tolist()
                # head-level structure: variance of the 8 head means over the variance of all rows
                e[f"{nm}_head_var_share"] = float(hh.var() / h.var()) if h.var() > 0 else float("nan")
            for nm, h in (("q", q_c), ("k", k_c)):
                e[f"{nm}_pair_cum"] = pair_profile(h).tolist()
            e["q_row__k_row"] = float(np.corrcoef(q_r, k_r)[0, 1])
            e["q_pair__k_pair"] = float(np.corrcoef(pair_profile(q_r), pair_profile(k_r))[0, 1])
            out.append(e)
        return out
    qk = {arm: {str(s): qk_struct(arm, s) for s in common} for arm in ("base", "nope")}

    # pseudo-pair (NoPE) vs true-pair (base) profile correlation at 30k, static and cumulative, per layer; permutation null on 32 points
    def pair_corr(key):
        rows = []
        for L in range(6):
            b = np.array(qk["base"]["30000"][L][key]); n = np.array(qk["nope"]["30000"][L][key])
            r = float(np.corrcoef(b, n)[0, 1])
            nulls = np.array([np.corrcoef(b, RNG.permutation(n))[0, 1] for _ in range(NS["N_NULL"])])
            rows.append({"layer": L, "r": r, "null_abs_p95": float(np.quantile(np.abs(nulls), 0.95))})
        return rows
    pair_cmp = {f"{nm}_{kind}": pair_corr(f"{nm}_pair_{kind}") for nm in ("q", "k") for kind in ("static", "cum")}
    # base seed-internal reference for the same statistic: cross-seed pair-profile correlation (S1, cumulative 0->30k)
    # (base 30k vs base 30k is 1 by construction; the informative comparison is nope-vs-base against the cross-seed band)

    # ---------------- (3) S10 transfer, (4) S13 gain/balance ----------------
    def paths(arm, s):
        out = []
        for L in range(6):
            m = lay[(arm, s)][L]
            g_r, u_r, d_c = prof(m["gate_proj"], 1), prof(m["up_proj"], 1), prof(m["down_proj"], 0)
            v_r, o_c = prof(m["v_proj"], 1), prof(m["o_proj"], 0)
            qp, kp = pair_profile(prof(m["q_proj"], 1)), pair_profile(prof(m["k_proj"], 1))
            e = {"layer": L,
                 "gate_row__down_col": float(np.corrcoef(g_r, d_c)[0, 1]), "up_row__down_col": float(np.corrcoef(u_r, d_c)[0, 1]),
                 "v_row__o_col": float(np.corrcoef(v_r, o_c)[0, 1])}
            for nm, (a, b) in (("UD", (u_r, d_c)), ("VO", (v_r, o_c)), ("QK", (qp, kp))):
                e[nm] = gb_stats(a, b); e[nm]["null_ratio_q025_med_q975"] = null_ratio(a, b, 100)
            out.append(e)
        return out
    pth = {arm: {str(s): paths(arm, s) for s in common} for arm in ("base", "nope")}

    def pmed(arm, s, key, sub=None):
        v = [(e[key][sub] if sub else e[key]) for e in pth[arm][str(s)]]; return float(np.median(v))

    # ---------------- (5) loss ----------------
    loss = {"base": loss_curve(BASE), "nope": loss_curve(NOPE)}
    loss30 = {a: [l for s, l in loss[a] if s == 30000][-1] for a in loss}

    # ---------------- criteria (descriptive) ----------------
    crit = {
        "H_ident_30k_nope_over_base": {LABEL[k]: H_ident["nope"][k]["30000"] / H_ident["base"][k]["30000"] for k in KINDS},
        "qk_H_row_30k": {a: {LABEL[k]: H_ident[a][k]["30000"] for k in ("q_proj", "k_proj")} for a in ("base", "nope")},
        "pseudo_pair_vs_true_pair_corr_30k": {key: {"median_r": float(np.median([r["r"] for r in rows])), "range": [min(r["r"] for r in rows), max(r["r"] for r in rows)],
                                                    "null_abs_p95_median": float(np.median([r["null_abs_p95"] for r in rows])),
                                                    "n_layers_above_null": int(sum(abs(r["r"]) > r["null_abs_p95"] for r in rows))} for key, rows in pair_cmp.items()},
        "cross_seed_reference_pair_corr": REF_CROSS,
        "head_var_share_30k": {a: {nm: float(np.median([e[f"{nm}_head_var_share"] for e in qk[a]["30000"]])) for nm in ("q", "k")} for a in ("base", "nope")},
        "q_k_corr_30k": {a: {"row": float(np.median([e["q_row__k_row"] for e in qk[a]["30000"]])), "pair": float(np.median([e["q_pair__k_pair"] for e in qk[a]["30000"]]))} for a in ("base", "nope")},
        "transfer_30k": {a: {key: pmed(a, 30000, key) for key in ("gate_row__down_col", "up_row__down_col", "v_row__o_col")} for a in ("base", "nope")},
        "gain_balance_ratio_30k": {a: {p: {"ratio": pmed(a, 30000, p, "ratio"), "corr": pmed(a, 30000, p, "corr"),
                                          "null_q975": float(np.median([e[p]["null_ratio_q025_med_q975"][2] for e in pth[a]["30000"]]))} for p in ("UD", "VO", "QK")} for a in ("base", "nope")},
        "loss_30k": loss30,
    }

    res = {"script": Path(__file__).name, "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "protocol_segments_sha16": {k: v for k, v in NS.items() if k.startswith("__seg_sha16__")},
           "runs": {"nope": str(NOPE.relative_to(ROOT)), "base": str(BASE.relative_to(ROOT))}, "nope_run_meta": {k: meta_n[k] for k in ("arm", "seed", "steps", "sigma", "lwd", "bs", "seq") if k in meta_n},
           "common_steps": common, "heads": HEADS, "head_dim": HEAD_DIM, "n_fits": len(fits),
           "criteria": crit, "H_identity_axis_by_step": H_ident, "k_and_H_by_step": K_tab,
           "qk_structure": qk, "pair_profile_comparison_30k": pair_cmp, "paths": pth, "loss": loss,
           "fits": [{k: e[k] for k in ("arm", "step", "layer", "kind", "k_raw", "k_row", "k_col", "k_bi", "H_row", "H_col", "r2_raw", "r2_bi")} for e in fits]}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=1)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "legend.fontsize": 8})
    fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.2))
    xs = np.array(common, dtype=float)
    def _sx(v):
        v = np.asarray(v, dtype=float); return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))
    ax = axs[0]
    for arm, c in (("base", C_BASE), ("nope", C_NOPE)):
        for k, ls in (("q_proj", "-"), ("k_proj", "--")):
            ax.plot(_sx(xs), [H_ident[arm][k][str(s)] for s in common], color=c, ls=ls, lw=1.8, marker="o", ms=3.4)
    ax.set_xticks(_sx([0, 800, 3200, 10000, 30000])); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
    ax.set_xlabel("training step (linear to 800, log above)"); ax.set_ylabel(r"$H_{\mathrm{row}}$ of $q$ (solid), $k$ (dashed)")
    ax.set_title("(a) q/k row-scale spread, RoPE vs NoPE", loc="left")
    ax.legend(handles=[Line2D([], [], color=C_BASE, lw=1.8, label="base (RoPE)"), Line2D([], [], color=C_NOPE, lw=1.8, label="NoPE")], frameon=False)
    ax = axs[1]
    for arm, c in (("base", C_BASE), ("nope", C_NOPE)):
        for nm, ls in (("q", "-"), ("k", "--")):
            P = np.median([e[f"{nm}_pair_static"] for e in qk[arm]["30000"]], 0)
            ax.plot(range(HEAD_DIM // 2), P, color=c, ls=ls, lw=1.6, marker="o", ms=2.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("rotary pair index (NoPE: same coordinate label, no frequency)"); ax.set_ylabel("row-scale profile at 30k (median over heads, layers)")
    ax.set_title("(b) Pair-indexed profile of q (solid) and k (dashed)", loc="left")
    ax = axs[2]
    w = 0.38; idx = np.arange(len(KINDS))
    ax.bar(idx - w / 2, [H_ident["base"][k]["30000"] for k in KINDS], w, color=C_BASE, label="base (RoPE)")
    ax.bar(idx + w / 2, [H_ident["nope"][k]["30000"] for k in KINDS], w, color=C_NOPE, label="NoPE")
    ax.set_xticks(idx); ax.set_xticklabels([LABEL[k] for k in KINDS]); ax.set_ylabel(r"identity-axis $H^W$ at 30k")
    ax.set_title("(c) Identity-axis spread per kind at 30k", loc="left"); ax.legend(frameon=False)
    ax = axs[3]
    P3 = ["UD", "VO", "QK"]; idx = np.arange(3)
    for i, (arm, c) in enumerate((("base", C_BASE), ("nope", C_NOPE))):
        vals = [crit["gain_balance_ratio_30k"][arm][p]["ratio"] for p in P3]
        nul = [crit["gain_balance_ratio_30k"][arm][p]["null_q975"] for p in P3]
        ax.bar(idx + (i - 0.5) * w, vals, w, color=c, label=arm if arm == "NoPE" else ("base (RoPE)" if arm == "base" else "NoPE"))
        ax.scatter(idx + (i - 0.5) * w, nul, marker="_", color="k", s=120, zorder=5)
    ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels(["up/down", "v/o", "q/k (pair)"]); ax.set_ylabel("var(g)/var(b) at 30k (tick: re-pairing null 97.5%)")
    ax.set_title("(d) Path gain/balance ratio, RoPE vs NoPE", loc="left"); ax.legend(frameon=False)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT_JSON, OUT_PNG)
    print(json.dumps(crit, indent=1)[:6000])


if __name__ == "__main__":
    main()
