"""Paper#5 v2 Figure S13: path-gain and balance modes of paired scale fields (zero GPU).

For two adjacent projections that share a unit identity, with centred log-scale profiles
h1, h2 (ln RMS along the identity axis minus its median, raw W, no normalization; same convention
as p5v2_identity_transfer_v1.py), define per unit
    gain     g = h1 + h2      (log of the product of the two scales: path gain)
    balance  b = h1 - h2      (log of the ratio: how the two sides share the scale)
Pairings:
    QK_pair : q rows vs k rows aggregated by RoPE pair (rotate_half pairs (j, j+hd/2), median over heads)
    QK_row  : q rows vs k rows per row (512)
    VO      : v rows vs o columns (512, same index);  VO_head: median over head_dim (8 points)
    UD      : up rows vs down columns (hidden units);  controls GD (gate/down), GU (gate/up)
Per layer / run / step: var(g), var(b), var(g)/var(b), corr(h1,h2).  Since
var(g)/var(b) = (v1+v2+2c)/(v1+v2-2c), a matched identity shows up as a ratio away from 1; under a
within-layer random re-pairing (null) or an adjacent-layer mismatch the ratio sits near 1.
Stability: correlation of g (and of b) across seeds / initialization families, at the unit index
(meaningful only where the index is an architectural coordinate: RoPE pair, head) and at the level
of the sorted values (distribution shape, meaningful for hidden units and channels whose index is
a permutation-symmetric label).

Sources (read-only):
  04_tclock_runs/wt_ckpts/tclock_D{1..4}_s{1,2,3}/step{800,1600,3200,5000}.pt        (48)
  04_ea_runs/ckpt_key/ea_{gaussian,laplace,uniform}_s{1,2,3}/ckpt/step{0,...,30000}.pt (63)
  05_ivn_runs/ivn_{base,ropeperm}/ckpt/step30000.pt                                   (2)
  08_paper5_draft/data/p5v2_identity_transfer_v1.json   (S10 correlations, cross-checked)
Outputs:
  08_paper5_draft/data/p5v2_path_gain_balance_v1.json
  08_paper5_draft/figures/P5v2_S13_path_gain_balance.png
"""
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_path_gain_balance_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S13_path_gain_balance.png"
S10 = ROOT / "08_paper5_draft/data/p5v2_identity_transfer_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

ARMS = ["D1", "D2", "D3", "D4"]; SEEDS = [1, 2, 3]; STEPS_T = [800, 1600, 3200, 5000]
INITS = ["gaussian", "laplace", "uniform"]; STEPS_E = [0, 800, 3200, 10000, 20000, 25000, 30000]
N_NULL = 200
RNG = np.random.default_rng(0)
GROUPS = ["QK_pair", "QK_row", "VO", "VO_head", "UD", "GD", "GU"]
MAIN = ["QK_pair", "VO", "UD"]
GCOL = {"QK_pair": "#B2182B", "VO": "#2166AC", "UD": "#1b9e77", "GD": "#7570b3", "GU": "#e7298a", "QK_row": "#e08080", "VO_head": "#80a0d0"}
ARMCOL = {"D1": "#1b9e77", "D2": "#7570b3", "D3": "#d95f02", "D4": "#e7298a"}
INITCOL = {"gaussian": "0.25", "laplace": "#B2182B", "uniform": "#2166AC"}


def ckpt_list():
    L = []
    for a in ARMS:
        for s in SEEDS:
            for st in STEPS_T:
                L.append(("tclock", f"{a}_s{s}", st, ROOT / f"04_tclock_runs/wt_ckpts/tclock_{a}_s{s}/step{st}.pt"))
    for i in INITS:
        for s in SEEDS:
            for st in STEPS_E:
                L.append(("ea", f"{i}_s{s}", st, ROOT / f"04_ea_runs/ckpt_key/ea_{i}_s{s}/ckpt/step{st}.pt"))
    for arm in ("base", "ropeperm"):
        L.append(("ivn", arm, 30000, ROOT / f"05_ivn_runs/ivn_{arm}/ckpt/step30000.pt"))
    return L


CK = ckpt_list()
assert len(CK) == 48 + 63 + 2, len(CK)
missing = [str(p) for *_, p in CK if not p.exists()]
assert not missing, missing

IVN = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
HEADS = int(IVN["heads"]); HEAD_DIM = int(IVN["head_dim"])
S10J = json.load(open(S10))
S10ROWS = {(r["family"], r["run"], r["step"], r["layer"]): r for r in S10J["rows"]}


def prof(x, axis):
    r = np.sqrt((x ** 2).mean(axis)); lr = np.log(r); return lr - np.median(lr)


def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return lay


def pair_profile(h_row):
    half = HEAD_DIM // 2
    m = h_row.reshape(HEADS, HEAD_DIM)
    return np.median((m[:, :half] + m[:, half:]) / 2, 0)


def head_profile(h):
    return np.median(h.reshape(HEADS, HEAD_DIM), 1)


def gb_stats(h1, h2):
    g = h1 + h2; b = h1 - h2
    vg, vb = float(g.var()), float(b.var())
    return {"var_g": vg, "var_b": vb, "ratio": vg / vb, "corr": float(np.corrcoef(h1, h2)[0, 1])}


def null_ratio(h1, h2, n=N_NULL):
    r = []
    for _ in range(n):
        p = RNG.permutation(h2); r.append(float((h1 + p).var() / (h1 - p).var()))
    r = np.array(r); return [float(np.quantile(r, 0.025)), float(np.median(r)), float(np.quantile(r, 0.975))]


def profiles(m):
    q_r = prof(m["q_proj"], 1); k_r = prof(m["k_proj"], 1)
    v_r = prof(m["v_proj"], 1); o_c = prof(m["o_proj"], 0)
    u_r = prof(m["up_proj"], 1); g_r = prof(m["gate_proj"], 1); d_c = prof(m["down_proj"], 0)
    assert q_r.shape == k_r.shape == v_r.shape == o_c.shape and u_r.shape == g_r.shape == d_c.shape
    return {"QK_pair": (pair_profile(q_r), pair_profile(k_r)), "QK_row": (q_r, k_r),
            "VO": (v_r, o_c), "VO_head": (head_profile(v_r), head_profile(o_c)),
            "UD": (u_r, d_c), "GD": (g_r, d_c), "GU": (g_r, u_r)}


rows = []; vec = {}   # vec[(fam, run, step, layer, group)] = (g, b)
example = None
for fam, run, step, p in CK:
    lay = load_layers(p); Ls = sorted(lay); assert len(Ls) == 6, (p, len(Ls))
    P = {L: profiles(lay[L]) for L in Ls}
    for L in Ls:
        e = {"family": fam, "run": run, "step": step, "layer": L}
        for gname in GROUPS:
            h1, h2 = P[L][gname]
            st = gb_stats(h1, h2); e[gname] = st
            vec[(fam, run, step, L, gname)] = (h1 + h2, h1 - h2)
            if gname in MAIN:
                e[gname]["null_ratio_q025_med_q975"] = null_ratio(h1, h2)
                Ln = Ls[(Ls.index(L) + 1) % len(Ls)]
                h2n = P[Ln][gname][1]
                e[gname]["mismatch_next_layer"] = gb_stats(h1, h2n)
        # cross-check against S10 correlations where that record exists
        s10 = S10ROWS.get((fam, run, step, L))
        if s10 is not None:
            for gname, key in (("UD", "up_row__down_col"), ("GD", "gate_row__down_col"), ("GU", "gate_row__up_row"),
                               ("VO", "v_row__o_col"), ("VO_head", "v_head__o_head"), ("QK_row", "q_row__k_row"), ("QK_pair", "q_pair__k_pair")):
                assert abs(e[gname]["corr"] - s10[key][0]) < 1e-9, (fam, run, step, L, gname, e[gname]["corr"], s10[key][0])
            e["s10_crosschecked"] = True
        rows.append(e)
        if fam == "tclock" and run == "D1_s1" and step == 5000 and L == 3:
            example = {"h_up": P[L]["UD"][0].tolist(), "h_down": P[L]["UD"][1].tolist(), "layer": L, "run": run, "step": step}
    print(fam, run, step, "done", flush=True)
assert len(rows) == len(CK) * 6, len(rows)
n_cc = sum(1 for r in rows if r.get("s10_crosschecked"))
n_expect = sum(1 for k in S10ROWS if any((f, r, st) == k[:3] for f, r, st, _ in CK))
assert n_cc == n_expect, (n_cc, n_expect)  # S10 also holds IVN step 0, which is not loaded here


def med(vals):
    v = np.array(vals, dtype=float); return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)), "min": float(v.min()), "max": float(v.max())}


def sel(fam=None, step=None, arm=None, init=None, run=None):
    out = []
    for r in rows:
        if fam and r["family"] != fam: continue
        if step is not None and r["step"] != step: continue
        if arm and not r["run"].startswith(arm): continue
        if init and not r["run"].startswith(init): continue
        if run and r["run"] != run: continue
        out.append(r)
    return out


def block(rs, groups=GROUPS):
    d = {}
    for g in groups:
        d[g] = {k: med([r[g][k] for r in rs]) for k in ("var_g", "var_b", "ratio", "corr")}
        if g in MAIN:
            d[g]["null_ratio_q975_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][2] for r in rs]))
            d[g]["null_ratio_q025_median"] = float(np.median([r[g]["null_ratio_q025_med_q975"][0] for r in rs]))
            d[g]["mismatch_ratio"] = med([r[g]["mismatch_next_layer"]["ratio"] for r in rs])
            d[g]["mismatch_corr"] = med([r[g]["mismatch_next_layer"]["corr"] for r in rs])
    return d


# ---- stability of g and b across runs (index level and sorted level)
def stab(keys_a, keys_b):
    out = {}
    for g in GROUPS:
        gi, bi, gs, bs = [], [], [], []
        for ka, kb in zip(keys_a, keys_b):
            (ga, ba), (gb_, bb) = vec[ka + (g,)], vec[kb + (g,)]
            gi.append(np.corrcoef(ga, gb_)[0, 1]); bi.append(np.corrcoef(ba, bb)[0, 1])
            gs.append(np.corrcoef(np.sort(ga), np.sort(gb_))[0, 1]); bs.append(np.corrcoef(np.sort(ba), np.sort(bb))[0, 1])
        out[g] = {"g_index": med(gi), "b_index": med(bi), "g_sorted": med(gs), "b_sorted": med(bs)}
    return out


ka, kb = [], []
for a in ARMS:
    for st in STEPS_T:
        for L in range(6):
            for s1, s2 in combinations(SEEDS, 2):
                ka.append(("tclock", f"{a}_s{s1}", st, L)); kb.append(("tclock", f"{a}_s{s2}", st, L))
stab_tclock_seed = stab(ka, kb)
ka, kb = [], []
for st in STEPS_T:
    for L in range(6):
        for a in ARMS:
            for s1, s2 in combinations(SEEDS, 2):
                ka.append(("tclock", f"{a}_s{s1}", st, L)); kb.append(("tclock", f"{a}_s{s2}", st, L))
ka5 = [k for k in ka if k[2] == 5000]; kb5 = [k for k in kb if k[2] == 5000]
stab_tclock_seed_5000 = stab(ka5, kb5)
runs_e = [f"{i}_s{s}" for i in INITS for s in SEEDS]
ka, kb, ka_x, kb_x = [], [], [], []
for L in range(6):
    for r1, r2 in combinations(runs_e, 2):
        same = r1.split("_s")[0] == r2.split("_s")[0]
        (ka if same else ka_x).append(("ea", r1, 30000, L)); (kb if same else kb_x).append(("ea", r2, 30000, L))
stab_ea_same_init = stab(ka, kb); stab_ea_cross_init = stab(ka_x, kb_x)

summary = {
    "tclock_step5000": block(sel("tclock", 5000)),
    "tclock_by_step": {str(s): block(sel("tclock", s), MAIN + ["GD", "GU"]) for s in STEPS_T},
    "tclock_by_arm_step5000": {a: block(sel("tclock", 5000, arm=a), MAIN + ["GD", "GU"]) for a in ARMS},
    "ea_by_step_all_inits": {str(s): block(sel("ea", s), MAIN + ["GD", "GU"]) for s in STEPS_E},
    "ea_by_init_step30000": {i: block(sel("ea", 30000, init=i), MAIN + ["GD", "GU"]) for i in INITS},
    "ivn_step30000": {arm: block(sel("ivn", run=arm), MAIN + ["GD", "GU"]) for arm in ("base", "ropeperm")},
    "stability_tclock_cross_seed_all_steps": stab_tclock_seed,
    "stability_tclock_cross_seed_step5000": stab_tclock_seed_5000,
    "stability_ea_30000_same_init_cross_seed": stab_ea_same_init,
    "stability_ea_30000_cross_init": stab_ea_cross_init,
}

json.dump({"schema": "p5v2-path-gain-balance-v1", "script": Path(__file__).name, "script_sha256": SHA,
           "protocol": "raw W; h = ln RMS along the identity axis minus its median; g = h1+h2, b = h1-h2; ratio = var(g)/var(b); null = %d within-layer re-pairings of h2; mismatch = h2 of the next layer (wrap); stability = Pearson of g (b) between runs at the unit index and between sorted values" % N_NULL,
           "heads": HEADS, "head_dim": HEAD_DIM, "groups": GROUPS, "n_ckpt": len(CK), "n_rows": len(rows), "n_s10_crosschecked_rows": n_cc,
           "summary": summary, "example_D1_s1_step5000_L3_UD": example, "rows": rows}, open(OUT_JSON, "w"), indent=1)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})
fig = plt.figure(figsize=(14.0, 4.3)); gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 0.9], wspace=0.38)


def _sx(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))


# (a) ratio vs step
ax = fig.add_subplot(gs[0, 0])
for g in MAIN:
    ye = [summary["ea_by_step_all_inits"][str(s)][g]["ratio"]["median"] for s in STEPS_E]
    ax.plot(_sx(STEPS_E), ye, color=GCOL[g], lw=2.0, marker="o", ms=3.5, mec="white", zorder=4)
    yt = [summary["tclock_by_step"][str(s)][g]["ratio"]["median"] for s in STEPS_T]
    ax.plot(_sx(STEPS_T), yt, color=GCOL[g], lw=1.4, ls=(0, (3, 2)), marker="s", ms=3.2, mfc="white", zorder=3)
    nl = summary["tclock_step5000"][g]["null_ratio_q025_median"]; nh = summary["tclock_step5000"][g]["null_ratio_q975_median"]
    ax.axhspan(nl, nh, color="0.88", zorder=1)
    mm = summary["tclock_step5000"][g]["mismatch_ratio"]["median"]
    ax.scatter([_sx(5000)], [mm], marker="x", color=GCOL[g], s=28, zorder=5)
ax.axhline(1.0, color="0.4", lw=0.8)
xt = [0, 800, 3200, 10000, 30000]; ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlabel("training step  (linear to 800, log above)"); ax.set_ylabel("var(g) / var(b)   (g = h1+h2, b = h1−h2)")
ax.set_yscale("log")
ax.set_title("(a) var(g)/var(b) over training", loc="left", pad=6)
h = [Line2D([], [], color=GCOL[g], lw=2, label={"QK_pair": "q/k by RoPE pair", "VO": "v rows / o columns", "UD": "up rows / down columns"}[g]) for g in MAIN] + \
    [Line2D([], [], color="0.4", lw=2, label="E-A (median over 9 runs)"), Line2D([], [], color="0.4", lw=1.4, ls=(0, (3, 2)), marker="s", mfc="white", label="TCLOCK (median over 12 runs)"),
     Line2D([], [], marker="x", ls="", color="0.4", label="next-layer mismatch, 5,000"), Line2D([], [], color="0.88", lw=6, label="within-layer re-pairing null (95%)")]
ax.legend(handles=h, fontsize=6.6, frameon=False, loc="lower right", handlelength=1.6, handletextpad=0.4, borderpad=0.2)

# (b) stability of g and b
ax = fig.add_subplot(gs[0, 1])
labels = []; x = 0
for g, lab in (("QK_pair", "q/k pair\n(index)"), ("VO_head", "v/o head\n(index)"), ("UD", "up/down\n(sorted)"), ("VO", "v/o\n(sorted)")):
    lvl = "index" if "index" in lab else "sorted"
    for src, mk in (("stability_tclock_cross_seed_step5000", "o"), ("stability_ea_30000_cross_init", "^")):
        sg = summary[src][g][f"g_{lvl}"]; sb = summary[src][g][f"b_{lvl}"]
        ax.errorbar(x - 0.15, sg["median"], yerr=[[sg["median"] - sg["q25"]], [sg["q75"] - sg["median"]]], fmt=mk, color=GCOL[g], ms=5, capsize=2, mfc=GCOL[g] if mk == "o" else "white")
        ax.errorbar(x + 0.15, sb["median"], yerr=[[sb["median"] - sb["q25"]], [sb["q75"] - sb["median"]]], fmt=mk, color="0.45", ms=5, capsize=2, mfc="0.45" if mk == "o" else "white")
    labels.append(lab); x += 1
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color="0.4", lw=0.8); ax.set_ylim(-0.3, 1.05)
ax.set_ylabel("correlation between runs (median, IQR)")
ax.set_title("(b) g (colour) and b (grey) across runs", loc="left", pad=6)
h = [Line2D([], [], marker="o", ls="", color="0.3", label="TCLOCK, cross-seed within arm, 5,000"), Line2D([], [], marker="^", ls="", color="0.3", mfc="white", label="E-A, cross-initialization, 30k")]
ax.legend(handles=h, fontsize=6.8, frameon=False, loc="lower left")

# (c) example scatter up vs down
ax = fig.add_subplot(gs[0, 2])
hu = np.array(example["h_up"]); hd = np.array(example["h_down"])
ax.scatter(hu, hd, s=5, color=GCOL["UD"], alpha=0.45)
lim = max(abs(hu).max(), abs(hd).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=1.0, ls=(0, (4, 2))); ax.plot([-lim, lim], [lim, -lim], color="0.6", lw=1.0, ls=(0, (1, 2)))
ax.text(lim * 0.55, lim * 0.75, "g", color="0.3", fontsize=9); ax.text(lim * 0.55, -lim * 0.85, "b", color="0.6", fontsize=9)
st = gb_stats(hu, hd)
ax.text(0.03, 0.97, "%s, step %d, layer %d, %d units\nvar(g)/var(b) = %.1f, r = %.2f" % (example["run"], example["step"], example["layer"], hu.size, st["ratio"], st["corr"]), transform=ax.transAxes, va="top", fontsize=7.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("up row profile  h"); ax.set_ylabel("down column profile  h")
ax.set_title("(c) up vs down, one layer", loc="left", pad=6)

# (d) arms
ax = fig.add_subplot(gs[0, 3])
w = 0.2
for gi, g in enumerate(MAIN):
    for ai, a in enumerate(ARMS):
        v = summary["tclock_by_arm_step5000"][a][g]["ratio"]
        ax.bar(ai + (gi - 1) * w, v["median"], width=w * 0.9, color=GCOL[g], alpha=0.85)
        ax.plot([ai + (gi - 1) * w] * 2, [v["q25"], v["q75"]], color="0.2", lw=0.9)
ax.axhline(1.0, color="0.4", lw=0.8); ax.set_yscale("log")
ax.set_xticks(range(4)); ax.set_xticklabels(ARMS); ax.set_xlabel("data arm (step 5,000)")
ax.set_ylabel("var(g) / var(b)")
ax.set_title("(d) By data arm", loc="left", pad=6)
h = [Line2D([], [], color=GCOL[g], lw=6, label={"QK_pair": "q/k pair", "VO": "v/o", "UD": "up/down"}[g]) for g in MAIN]
ax.legend(handles=h, fontsize=7, frameon=False, loc="upper right")
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT_FIG, OUT_JSON)

# ------------------------------------------------------------------ console report
T5 = summary["tclock_step5000"]
for g in GROUPS:
    s = T5[g]; line = f"{g:8s} ratio {s['ratio']['median']:.2f} [{s['ratio']['min']:.2f},{s['ratio']['max']:.2f}] corr {s['corr']['median']:+.2f} var_g {s['var_g']['median']:.4f} var_b {s['var_b']['median']:.4f}"
    if g in MAIN: line += f" | null97.5 {s['null_ratio_q975_median']:.2f} mismatch {s['mismatch_ratio']['median']:.2f}"
    print(line)
for g in MAIN + ["GD", "GU"]:
    print(g, "tclock by step", [round(summary["tclock_by_step"][str(s)][g]["ratio"]["median"], 2) for s in STEPS_T],
          "var_g", [round(summary["tclock_by_step"][str(s)][g]["var_g"]["median"], 4) for s in STEPS_T],
          "var_b", [round(summary["tclock_by_step"][str(s)][g]["var_b"]["median"], 4) for s in STEPS_T])
    print(g, "ea by step ratio", [round(summary["ea_by_step_all_inits"][str(s)][g]["ratio"]["median"], 2) for s in STEPS_E],
          "var_g", [round(summary["ea_by_step_all_inits"][str(s)][g]["var_g"]["median"], 4) for s in STEPS_E],
          "var_b", [round(summary["ea_by_step_all_inits"][str(s)][g]["var_b"]["median"], 4) for s in STEPS_E])
    print(g, "arms", {a: round(summary["tclock_by_arm_step5000"][a][g]["ratio"]["median"], 2) for a in ARMS},
          "ivn", {arm: round(summary["ivn_step30000"][arm][g]["ratio"]["median"], 2) for arm in ("base", "ropeperm")},
          "ea inits", {i: round(summary["ea_by_init_step30000"][i][g]["ratio"]["median"], 2) for i in INITS})
for src in ("stability_tclock_cross_seed_step5000", "stability_ea_30000_same_init_cross_seed", "stability_ea_30000_cross_init"):
    print(src)
    for g in GROUPS:
        s = summary[src][g]; print(f"   {g:8s} g_idx {s['g_index']['median']:+.2f} b_idx {s['b_index']['median']:+.2f} g_sorted {s['g_sorted']['median']:+.2f} b_sorted {s['b_sorted']['median']:+.2f}")
