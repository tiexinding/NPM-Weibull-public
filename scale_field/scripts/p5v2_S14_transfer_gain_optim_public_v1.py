"""Paper#5 supplement S14: identity-axis transfer (S10) and path gain/balance (S13) extended to
  J1  the optimizer arms of the IVN family (base / nomom beta1=0 / flatv row-flat v-hat), 9 checkpoints each
  J2  the final public Pythia checkpoints at four sizes (70m/160m/410m/1b)

Same protocol as S10/S13: h = ln RMS along the axis minus its median (raw W, no normalization);
g = h1 + h2, b = h1 - h2; ratio = var(g)/var(b); null = within-layer random re-pairing of h2
(N_NULL draws, 2.5/97.5 pct); mismatch = h2 of the next layer (wrap).  H_row / H_col = sd of ln RMS.

Pythia: fused QKV split per head as in p5v2_public_knorm_refit_v1.py (reshape(heads, 3, hd, hidden)), so
v rows and o (dense) columns share the index h*hd + d; RoPE acts on the first rotary_pct*hd dims of each
head with rotate_half pairs (j, j + rot/2); the q/k pair profile uses those rows only.

Outputs: 08_paper5_draft/data/p5v2_S14_transfer_gain_optim_public_v1.json
         08_paper5_draft/figures/P5v2_S14_transfer_gain_optim_public.png
"""
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from safetensors import safe_open
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_S14_transfer_gain_optim_public_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S14_transfer_gain_optim_public.png"
S13 = json.load(open(ROOT / "08_paper5_draft/data/p5v2_path_gain_balance_v1.json"))
S10 = json.load(open(ROOT / "08_paper5_draft/data/p5v2_identity_transfer_v1.json"))
IVNJ = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
HUB = Path.home() / ".cache/huggingface/hub"

ARMS = ["base", "nomom", "flatv"]
STEPS_I = [0, 800, 3200, 5000, 10000, 15000, 20000, 25000, 30000]
N_NULL = 200
RNG = np.random.default_rng(0)
GROUPS = ["QK_pair", "QK_row", "VO", "VO_head", "UD", "GD", "GU"]
MAIN = ["QK_pair", "VO", "UD"]
PUB_GROUPS = ["QK_pair", "QK_row", "VO", "VO_head", "FIO"]   # FIO = ffn_in rows vs ffn_out columns
PUB_MAIN = ["QK_pair", "VO", "FIO"]
SIZES = {"70m": dict(hidden=512, heads=8, layers=6, ref="main"),
         "160m": dict(hidden=768, heads=12, layers=12, ref="step143000"),
         "410m": dict(hidden=1024, heads=16, layers=24, ref="main"),
         "1b": dict(hidden=2048, heads=8, layers=16, ref="main")}
HEADS = int(IVNJ["heads"]); HEAD_DIM = int(IVNJ["head_dim"])
ARMDEF = {}
for a in ARMS:
    m = json.load(open(ROOT / f"05_ivn_runs/ivn_{a}/run_meta.json"))
    ARMDEF[a] = {"beta1": m["beta1"], "beta2": m["beta2"], "arm": m["arm"]}
assert ARMDEF["nomom"]["beta1"] == 0.0 and ARMDEF["base"]["beta1"] == 0.9 and ARMDEF["flatv"]["beta1"] == 0.9, ARMDEF


# ----------------------------------------------------------------------------- shared functions (S10/S13 protocol)
def prof(x, axis):
    r = np.sqrt((x ** 2).mean(axis)); lr = np.log(r); return lr - np.median(lr)


def hsd(x, axis):
    return float(np.log(np.sqrt((x ** 2).mean(axis))).std())


def pcorr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def gb_stats(h1, h2):
    g = h1 + h2; b = h1 - h2
    vg, vb = float(g.var()), float(b.var())
    return {"var_g": vg, "var_b": vb, "ratio": vg / vb, "corr": pcorr(h1, h2), "spearman": float(spearmanr(h1, h2)[0])}


def null_ratio(h1, h2, n=N_NULL):
    r = []; c = []
    for _ in range(n):
        p = RNG.permutation(h2); r.append(float((h1 + p).var() / (h1 - p).var())); c.append(abs(pcorr(h1, p)))
    r = np.array(r); return [float(np.quantile(r, 0.025)), float(np.median(r)), float(np.quantile(r, 0.975))], float(np.quantile(c, 0.95))


def pair_profile(h_row, heads, hd, rot):
    """median over heads of the rotate_half pair mean, restricted to the rotary dims of each head."""
    m = h_row.reshape(heads, hd)[:, :rot]
    half = rot // 2
    return np.median((m[:, :half] + m[:, half:]) / 2, 0)


def head_profile(h, heads, hd):
    return np.median(h.reshape(heads, hd), 1)


# ----------------------------------------------------------------------------- J1: IVN optimizer arms
def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return lay


def llama_profiles(m):
    q_r = prof(m["q_proj"], 1); k_r = prof(m["k_proj"], 1)
    v_r = prof(m["v_proj"], 1); o_c = prof(m["o_proj"], 0)
    u_r = prof(m["up_proj"], 1); g_r = prof(m["gate_proj"], 1); d_c = prof(m["down_proj"], 0)
    assert q_r.shape == k_r.shape == v_r.shape == o_c.shape == (HEADS * HEAD_DIM,) and u_r.shape == g_r.shape == d_c.shape
    return {"QK_pair": (pair_profile(q_r, HEADS, HEAD_DIM, HEAD_DIM), pair_profile(k_r, HEADS, HEAD_DIM, HEAD_DIM)),
            "QK_row": (q_r, k_r), "VO": (v_r, o_c),
            "VO_head": (head_profile(v_r, HEADS, HEAD_DIM), head_profile(o_c, HEADS, HEAD_DIM)),
            "UD": (u_r, d_c), "GD": (g_r, d_c), "GU": (g_r, u_r)}


def llama_H(m):
    return {"q": hsd(m["q_proj"], 1), "k": hsd(m["k_proj"], 1), "v_row": hsd(m["v_proj"], 1), "v_col": hsd(m["v_proj"], 0),
            "o_col": hsd(m["o_proj"], 0), "gate": hsd(m["gate_proj"], 1), "up": hsd(m["up_proj"], 1), "down_col": hsd(m["down_proj"], 0)}


CK = [(a, st, ROOT / f"05_ivn_runs/ivn_{a}/ckpt/step{st}.pt") for a in ARMS for st in STEPS_I]
missing = [str(p) for *_, p in CK if not p.exists()]
assert not missing, missing
assert len(CK) == 27, len(CK)

S10ROWS = {(r["run"], r["step"], r["layer"]): r for r in S10["rows"] if r["family"] == "ivn"}
rows_i = []
for arm, step, p in CK:
    lay = load_layers(p); Ls = sorted(lay); assert len(Ls) == 6, (p, len(Ls))
    P = {L: llama_profiles(lay[L]) for L in Ls}
    for L in Ls:
        e = {"arm": arm, "step": step, "layer": L, "H": llama_H(lay[L])}
        for gname in GROUPS:
            h1, h2 = P[L][gname]; st = gb_stats(h1, h2); e[gname] = st
            if gname in MAIN:
                nr, nc = null_ratio(h1, h2); st["null_ratio_q025_med_q975"] = nr; st["null_abs_corr_p95"] = nc
                Ln = Ls[(Ls.index(L) + 1) % len(Ls)]
                st["mismatch_next_layer"] = gb_stats(h1, P[Ln][gname][1])
        s10 = S10ROWS.get((arm, step, L))
        if s10 is not None:  # S10 holds IVN base at steps 0 and 30000
            for gname, key in (("UD", "up_row__down_col"), ("GD", "gate_row__down_col"), ("GU", "gate_row__up_row"),
                               ("VO", "v_row__o_col"), ("VO_head", "v_head__o_head"), ("QK_row", "q_row__k_row"), ("QK_pair", "q_pair__k_pair")):
                assert abs(e[gname]["corr"] - s10[key][0]) < 1e-9, (arm, step, L, gname, e[gname]["corr"], s10[key][0])
            e["s10_crosschecked"] = True
        rows_i.append(e)
    print("ivn", arm, step, "done", flush=True)
assert len(rows_i) == 27 * 6
assert sum(1 for r in rows_i if r.get("s10_crosschecked")) == 12, "IVN base step 0 and 30000 x 6 layers must cross-check against S10"
# cross-check S13 medians for IVN base step 30000
for gname in [g for g in GROUPS if g in S13["summary"]["ivn_step30000"]["base"]]:
    v = np.median([r[gname]["ratio"] for r in rows_i if r["arm"] == "base" and r["step"] == 30000])
    ref = S13["summary"]["ivn_step30000"]["base"][gname]["ratio"]["median"]
    assert abs(v - ref) < 1e-9, (gname, v, ref)


def med(vals):
    v = np.array(vals, dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def block_i(rs, groups=GROUPS):
    d = {}
    for g in groups:
        d[g] = {k: med([r[g][k] for r in rs]) for k in ("var_g", "var_b", "ratio", "corr", "spearman")}
        if g in MAIN:
            d[g]["null_ratio_q975_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][2] for r in rs]))
            d[g]["null_ratio_q025_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][0] for r in rs]))
            d[g]["null_abs_corr_p95_median"] = float(np.median([r[g]["null_abs_corr_p95"] for r in rs]))
            d[g]["mismatch_ratio"] = med([r[g]["mismatch_next_layer"]["ratio"] for r in rs])
            d[g]["mismatch_corr"] = med([r[g]["mismatch_next_layer"]["corr"] for r in rs])
    d["H"] = {k: med([r["H"][k] for r in rs]) for k in rs[0]["H"]}
    return d


summ_i = {"by_arm_step": {a: {str(s): block_i([r for r in rows_i if r["arm"] == a and r["step"] == s]) for s in STEPS_I} for a in ARMS}}

# ----------------------------------------------------------------------------- J2: public Pythia
def st_path(model, ref):
    base = HUB / f"models--EleutherAI--{model}"
    snap = (base / "refs" / ref).read_text().strip()
    p = glob.glob(str(base / "snapshots" / snap / "*.safetensors")); assert p, (model, ref, snap)
    cfg = json.load(open(glob.glob(str(base / "snapshots" / snap / "config.json"))[0]))
    return p[0], cfg


def pythia_mats(path, hidden, heads, layers):
    hd = hidden // heads
    with safe_open(path, framework="np") as sf:
        for b in range(layers):
            pre = f"gpt_neox.layers.{b}"
            qkv = sf.get_tensor(f"{pre}.attention.query_key_value.weight").astype(np.float64)
            assert qkv.shape == (3 * hidden, hidden)
            w3 = qkv.reshape(heads, 3, hd, hidden)
            yield b, {"q": w3[:, 0].reshape(heads * hd, hidden), "k": w3[:, 1].reshape(heads * hd, hidden),
                      "v": w3[:, 2].reshape(heads * hd, hidden),
                      "o": sf.get_tensor(f"{pre}.attention.dense.weight").astype(np.float64),
                      "ffn_in": sf.get_tensor(f"{pre}.mlp.dense_h_to_4h.weight").astype(np.float64),
                      "ffn_out": sf.get_tensor(f"{pre}.mlp.dense_4h_to_h.weight").astype(np.float64)}


def pythia_profiles(m, heads, hd, rot):
    q_r = prof(m["q"], 1); k_r = prof(m["k"], 1); v_r = prof(m["v"], 1); o_c = prof(m["o"], 0)
    fi_r = prof(m["ffn_in"], 1); fo_c = prof(m["ffn_out"], 0)
    assert q_r.shape == k_r.shape == v_r.shape == o_c.shape == (heads * hd,) and fi_r.shape == fo_c.shape == (4 * heads * hd,)
    return {"QK_pair": (pair_profile(q_r, heads, hd, rot), pair_profile(k_r, heads, hd, rot)), "QK_row": (q_r, k_r),
            "VO": (v_r, o_c), "VO_head": (head_profile(v_r, heads, hd), head_profile(o_c, heads, hd)), "FIO": (fi_r, fo_c)}


rows_p = []; pub_meta = {}
for size, cfg in SIZES.items():
    path, hf = st_path(f"pythia-{size}", cfg["ref"])
    heads, hidden, layers = hf["num_attention_heads"], hf["hidden_size"], hf["num_hidden_layers"]
    assert (heads, hidden, layers) == (cfg["heads"], cfg["hidden"], cfg["layers"]), (size, heads, hidden, layers)
    hd = hidden // heads; rot = int(hd * hf["rotary_pct"]); assert rot % 2 == 0 and 0 < rot <= hd
    pub_meta[size] = {"heads": heads, "head_dim": hd, "layers": layers, "rotary_dims": rot, "n_pairs": rot // 2, "ref": cfg["ref"], "file": path}
    P = {}
    for b, m in pythia_mats(path, hidden, heads, layers):
        P[b] = (pythia_profiles(m, heads, hd, rot), {"q": hsd(m["q"], 1), "k": hsd(m["k"], 1), "v_row": hsd(m["v"], 1), "v_col": hsd(m["v"], 0),
                                                       "o_col": hsd(m["o"], 0), "ffn_in": hsd(m["ffn_in"], 1), "ffn_out_col": hsd(m["ffn_out"], 0)})
    Ls = sorted(P)
    for L in Ls:
        e = {"size": size, "layer": L, "H": P[L][1]}
        for gname in PUB_GROUPS:
            h1, h2 = P[L][0][gname]; st = gb_stats(h1, h2); e[gname] = st
            if gname in PUB_MAIN:
                nr, nc = null_ratio(h1, h2); st["null_ratio_q025_med_q975"] = nr; st["null_abs_corr_p95"] = nc
                Ln = Ls[(Ls.index(L) + 1) % len(Ls)]
                st["mismatch_next_layer"] = gb_stats(h1, P[Ln][0][gname][1])
        rows_p.append(e)
    print("pythia", size, "done", flush=True)
assert len(rows_p) == sum(c["layers"] for c in SIZES.values()) == 58, len(rows_p)
# self-check: v/o alignment is by construction; a random permutation of o columns must kill the correlation
for size in SIZES:
    rs = [r for r in rows_p if r["size"] == size]
    assert np.median([r["VO"]["null_abs_corr_p95"] for r in rs]) < 0.2, size
    assert np.median([r["FIO"]["null_abs_corr_p95"] for r in rs]) < 0.2, size


def block_p(rs, groups=PUB_GROUPS):
    d = {}
    for g in groups:
        d[g] = {k: med([r[g][k] for r in rs]) for k in ("var_g", "var_b", "ratio", "corr", "spearman")}
        if g in PUB_MAIN:
            d[g]["null_ratio_q975_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][2] for r in rs]))
            d[g]["null_ratio_q025_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][0] for r in rs]))
            d[g]["null_abs_corr_p95_median"] = float(np.median([r[g]["null_abs_corr_p95"] for r in rs]))
            d[g]["mismatch_ratio"] = med([r[g]["mismatch_next_layer"]["ratio"] for r in rs])
            d[g]["mismatch_corr"] = med([r[g]["mismatch_next_layer"]["corr"] for r in rs])
    d["H"] = {k: med([r["H"][k] for r in rs]) for k in rs[0]["H"]}
    return d


summ_p = {"by_size": {s: block_p([r for r in rows_p if r["size"] == s]) for s in SIZES}}

json.dump({"schema": "p5v2-S14-transfer-gain-optim-public-v1", "script": Path(__file__).name, "script_sha256": SHA,
           "protocol": "raw W; h = ln RMS along the axis minus median; g=h1+h2, b=h1-h2; ratio var(g)/var(b); null = %d within-layer re-pairings (ratio 2.5/97.5 pct, |corr| 95 pct); mismatch = next layer (wrap); H = sd ln RMS. Pythia QKV split per head (heads,3,hd,hidden); q/k pair profile on rotary dims only, rotate_half pairs (j, j+rot/2)" % N_NULL,
           "ivn": {"arms": ARMDEF, "steps": STEPS_I, "heads": HEADS, "head_dim": HEAD_DIM, "n_rows": len(rows_i),
                   "s10_crosschecked_rows": 12, "s13_base_30000_medians_crosschecked": True, "summary": summ_i, "rows": rows_i},
           "public_pythia": {"meta": pub_meta, "n_rows": len(rows_p), "summary": summ_p, "rows": rows_p}},
          open(OUT_JSON, "w"), indent=1)
print("wrote", OUT_JSON)

# ----------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
GCOL = {"QK_pair": "#B2182B", "VO": "#2166AC", "UD": "#1b9e77", "FIO": "#1b9e77"}
GLAB = {"QK_pair": "q/k by RoPE pair", "VO": "v rows / o columns", "UD": "up rows / down columns", "FIO": "ffn in rows / ffn out columns"}
ALS = {"base": "-", "nomom": (0, (4, 2)), "flatv": (0, (1, 1.5))}
fig = plt.figure(figsize=(16.0, 4.4)); gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 1.05, 0.85], wspace=0.34)


def _sx(v):
    v = np.asarray(v, dtype=float); return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))


xt = [0, 800, 3200, 10000, 30000]
# (a) correlations vs step, three arms
ax = fig.add_subplot(gs[0, 0])
for g in MAIN:
    for a in ARMS:
        y = [summ_i["by_arm_step"][a][str(s)][g]["corr"]["median"] for s in STEPS_I]
        ax.plot(_sx(STEPS_I), y, color=GCOL[g], ls=ALS[a], lw=1.8 if a == "base" else 1.4, marker="o", ms=2.8)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"]); ax.set_xlim(_sx(0) - 0.05, _sx(30000) + 0.05)
ax.set_ylim(-0.1, 1.02); ax.set_xlabel("training step  (linear to 800, log above)"); ax.set_ylabel("correlation, matched units (median over 6 layers)")
ax.set_title("(a) Transfer correlation, optimizer arms", loc="left")
h = [Line2D([], [], color=GCOL[g], lw=1.8, label=GLAB[g]) for g in MAIN] + [Line2D([], [], color="0.3", ls=ALS[a], lw=1.4, label={"base": "base (β₁=0.9)", "nomom": "nomom (β₁=0)", "flatv": "flatv (row-flat v̂)"}[a]) for a in ARMS]
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=7.4)
# (b) ratio vs step
ax = fig.add_subplot(gs[0, 1])
for g in MAIN:
    for a in ARMS:
        y = [summ_i["by_arm_step"][a][str(s)][g]["ratio"]["median"] for s in STEPS_I]
        ax.plot(_sx(STEPS_I), y, color=GCOL[g], ls=ALS[a], lw=1.8 if a == "base" else 1.4, marker="o", ms=2.8)
nlo = min(summ_i["by_arm_step"][a]["30000"][g]["null_ratio_q025_median"] for a in ARMS for g in ("VO", "UD"))
nhi = max(summ_i["by_arm_step"][a]["30000"][g]["null_ratio_q975_median"] for a in ARMS for g in ("VO", "UD"))
nqk = max(summ_i["by_arm_step"][a]["30000"]["QK_pair"]["null_ratio_q975_median"] for a in ARMS)
ax.axhspan(nlo, nhi, color="0.88", zorder=0); ax.axhline(1, color="0.5", lw=0.8)
ax.set_yscale("log"); ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"]); ax.set_xlim(_sx(0) - 0.05, _sx(30000) + 0.05)
ax.set_xlabel("training step  (linear to 800, log above)"); ax.set_ylabel("var(g) / var(b)")
ax.set_title("(b) var(g)/var(b), optimizer arms", loc="left")
ax.axhline(nqk, color=GCOL["QK_pair"], lw=0.8, ls=":")
ax.text(0.03, 0.97, "grey: re-pairing null (95%%) for v/o and up/down at 30k;\ndotted red: null 97.5%% for q/k pairs (32 points) = %.1f" % nqk, transform=ax.transAxes, va="top", fontsize=7.2, color="0.35")
# (c) public: correlations per layer per size
ax = fig.add_subplot(gs[0, 2])
SZ = list(SIZES); xpos = {s: i for i, s in enumerate(SZ)}
for g in PUB_MAIN:
    off = {"QK_pair": -0.22, "VO": 0.0, "FIO": 0.22}[g]
    for s in SZ:
        rs = [r for r in rows_p if r["size"] == s]
        ax.scatter([xpos[s] + off] * len(rs), [r[g]["corr"] for r in rs], s=9, color=GCOL[g], alpha=0.45, lw=0)
        ax.plot([xpos[s] + off - 0.09, xpos[s] + off + 0.09], [summ_p["by_size"][s][g]["corr"]["median"]] * 2, color=GCOL[g], lw=2.2)
        ax.scatter([xpos[s] + off], [summ_p["by_size"][s][g]["mismatch_corr"]["median"]], marker="x", s=18, color="0.4", zorder=4)
ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(range(len(SZ))); ax.set_xticklabels([s.upper() for s in SZ], fontsize=8.5); ax.set_xlabel("public Pythia checkpoint")
ax.set_ylim(-0.3, 1.02); ax.set_ylabel("correlation, matched units (per layer)")
ax.set_title("(c) Transfer correlation, public Pythia", loc="left")
h = [Line2D([], [], color=GCOL[g], lw=2.2, label=GLAB[g]) for g in PUB_MAIN] + [Line2D([], [], marker="x", ls="", color="0.4", label="next-layer mismatch (median)")]
ax.legend(handles=h, loc="lower left", frameon=False, fontsize=7.2)
# (d) public: ratio per size
ax = fig.add_subplot(gs[0, 3])
w = 0.26
for j, g in enumerate(PUB_MAIN):
    ys = [summ_p["by_size"][s][g]["ratio"]["median"] for s in SZ]
    lo = [summ_p["by_size"][s][g]["ratio"]["min"] for s in SZ]; hi = [summ_p["by_size"][s][g]["ratio"]["max"] for s in SZ]
    xs = np.arange(len(SZ)) + (j - 1) * w
    ax.bar(xs, ys, width=w, color=GCOL[g], alpha=0.9)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt="none", ecolor="0.2", elinewidth=0.8, capsize=0)
nlo = min(summ_p["by_size"][s][g]["null_ratio_q025_median"] for s in SZ for g in ("VO", "FIO"))
nhi = max(summ_p["by_size"][s][g]["null_ratio_q975_median"] for s in SZ for g in ("VO", "FIO"))
nqk_p = {s: summ_p["by_size"][s]["QK_pair"]["null_ratio_q975_median"] for s in SZ}
ax.axhspan(nlo, nhi, color="0.88", zorder=0); ax.axhline(1, color="0.5", lw=0.8)
for i, s in enumerate(SZ):
    ax.plot([i - 1.5 * w, i - 0.5 * w], [nqk_p[s]] * 2, color=GCOL["QK_pair"], lw=0.9, ls=":")
ax.text(0.03, 0.97, "grey: re-pairing null (95%), v/o and ffn;\ndotted: q/k-pair null 97.5%", transform=ax.transAxes, va="top", fontsize=7.0, color="0.35")
ax.set_yscale("log"); ax.set_xticks(range(len(SZ))); ax.set_xticklabels([s.upper() for s in SZ], fontsize=8.5)
ax.set_ylabel("var(g) / var(b)  (median; bar = layer range)"); ax.set_title("(d) var(g)/var(b), public Pythia", loc="left")
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT_FIG)

# ----------------------------------------------------------------------------- console summary
for a in ARMS:
    for s in (0, 800, 5000, 30000):
        b = summ_i["by_arm_step"][a][str(s)]
        print(a, s, " ".join(f"{g}: r {b[g]['corr']['median']:+.2f} ratio {b[g]['ratio']['median']:.1f}" for g in MAIN),
              "| H q %.3f k %.3f v_row %.3f v_col %.3f o_col %.3f gate %.3f up %.3f down_col %.3f" % tuple(b["H"][k]["median"] for k in ("q", "k", "v_row", "v_col", "o_col", "gate", "up", "down_col")))
    b = summ_i["by_arm_step"][a]["30000"]
    print(a, "30k null ratio 97.5", {g: round(b[g]["null_ratio_q975_median"], 2) for g in MAIN}, "mismatch", {g: round(b[g]["mismatch_ratio"]["median"], 2) for g in MAIN},
          "GD/GU", round(b["GD"]["corr"]["median"], 2), round(b["GU"]["corr"]["median"], 2), "ratios", round(b["GD"]["ratio"]["median"], 1), round(b["GU"]["ratio"]["median"], 1))
for s in SZ:
    b = summ_p["by_size"][s]
    print("pythia", s, pub_meta[s]["n_pairs"], "pairs |", " ".join(f"{g}: r {b[g]['corr']['median']:+.2f} [{b[g]['corr']['min']:+.2f},{b[g]['corr']['max']:+.2f}] ratio {b[g]['ratio']['median']:.1f} null97.5 {b[g]['null_ratio_q975_median']:.2f} mism {b[g]['mismatch_corr']['median']:+.2f}" for g in PUB_MAIN),
          "| QK_row r %.2f VO_head r %.2f" % (b["QK_row"]["corr"]["median"], b["VO_head"]["corr"]["median"]),
          "| H", {k: round(b["H"][k]["median"], 3) for k in b["H"]})
