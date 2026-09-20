"""Paper#5 v2 Figure S10: identity-axis transfer test (zero GPU).

Hypothesis under test: scale heterogeneity follows the functional identity of a feature,
not a fixed side of the matrix. Three transfer tests, per layer and per run:
  1. FFN hidden unit: row profile of gate/up (ln row-RMS, one per hidden unit) vs column
     profile of down (ln column-RMS, same hidden-unit index).  Also gate-row vs up-row.
  2. attention value channel: row profile of v (512 = heads x head_dim) vs column profile of o
     (same index); also head-level (median over head_dim, 8 points).
  3. q row profile vs k row profile (512), and aggregated by RoPE pair (rotate_half pairs
     (j, j+head_dim/2), median over heads; head_dim/2 points).
Controls: (i) permutation null (down column profile shuffled, 95th percentile over draws);
(ii) mismatch: gate row profile of layer L vs down column profile of layer L+1 (no shared unit);
(iii) step-0 sanity on the E-A and IVN runs (no field at initialization).

Profile convention (as in p5v2_twoaxis_bridge_v1.py): raw W, no normalization; ln RMS along the
axis, minus its median.  Correlations: Pearson and Spearman.

Sources (read-only):
  04_tclock_runs/wt_ckpts/tclock_D{1..4}_s{1,2,3}/step{800,1600,3200,5000}.pt   (48 ckpt)
  04_ea_runs/ckpt_key/ea_{gaussian,laplace,uniform}_s{1,2,3}/ckpt/step{0,30000}.pt  (18 ckpt)
  05_ivn_runs/ivn_base/ckpt/step{0,30000}.pt                                        (2 ckpt)
Outputs:
  08_paper5_draft/data/p5v2_identity_transfer_v1.json
  08_paper5_draft/figures/P5v2_S10_identity_transfer.png
"""
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_identity_transfer_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S10_identity_transfer.png"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

ARMS = ["D1", "D2", "D3", "D4"]; SEEDS = [1, 2, 3]; STEPS = [800, 1600, 3200, 5000]
INITS = ["gaussian", "laplace", "uniform"]
N_NULL = 200
RNG = np.random.default_rng(0)
C_SEL, C_TRA = "#B2182B", "#2166AC"


def ckpt_list():
    L = []
    for a in ARMS:
        for s in SEEDS:
            for st in STEPS:
                L.append(("tclock", f"{a}_s{s}", st, ROOT / f"04_tclock_runs/wt_ckpts/tclock_{a}_s{s}/step{st}.pt"))
    for i in INITS:
        for s in SEEDS:
            for st in (0, 30000):
                L.append(("ea", f"{i}_s{s}", st, ROOT / f"04_ea_runs/ckpt_key/ea_{i}_s{s}/ckpt/step{st}.pt"))
    for st in (0, 30000):
        L.append(("ivn", "base", st, ROOT / f"05_ivn_runs/ivn_base/ckpt/step{st}.pt"))
    return L


CK = ckpt_list()
assert len(CK) == 48 + 18 + 2, len(CK)
assert all(p.exists() for *_, p in CK), [str(p) for *_, p in CK if not p.exists()]


def prof(x, axis):
    r = np.sqrt((x ** 2).mean(axis)); lr = np.log(r); return lr - np.median(lr)


def corr2(x, y):
    return float(np.corrcoef(x, y)[0, 1]), float(spearmanr(x, y)[0])


def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return lay


def head_geom(q_shape, meta):
    d = q_shape[0]; H = meta["heads"]; hd = d // H
    assert hd * H == d, (d, H)
    return H, hd


def analyse_ckpt(fam, run, step, p, meta):
    lay = load_layers(p); Ls = sorted(lay)
    assert len(Ls) == 6, (p, len(Ls))
    H, hd = head_geom(lay[0]["q_proj"].shape, meta)
    out = []
    for L in Ls:
        m = lay[L]
        g_r = prof(m["gate_proj"], 1); u_r = prof(m["up_proj"], 1); d_c = prof(m["down_proj"], 0)
        assert g_r.shape == d_c.shape == u_r.shape, (g_r.shape, d_c.shape)
        v_r = prof(m["v_proj"], 1); o_c = prof(m["o_proj"], 0)
        q_r = prof(m["q_proj"], 1); k_r = prof(m["k_proj"], 1)
        assert v_r.shape == o_c.shape == q_r.shape == k_r.shape
        e = {"family": fam, "run": run, "step": step, "layer": L, "n_hidden": int(g_r.size), "n_model": int(v_r.size)}
        e["gate_row__down_col"] = corr2(g_r, d_c); e["up_row__down_col"] = corr2(u_r, d_c); e["gate_row__up_row"] = corr2(g_r, u_r)
        e["v_row__o_col"] = corr2(v_r, o_c)
        vh = np.median(v_r.reshape(H, hd), 1); oh = np.median(o_c.reshape(H, hd), 1)
        e["v_head__o_head"] = corr2(vh, oh)
        e["q_row__k_row"] = corr2(q_r, k_r)
        half = hd // 2
        qp = np.median((q_r.reshape(H, hd)[:, :half] + q_r.reshape(H, hd)[:, half:]) / 2, 0)
        kp = np.median((k_r.reshape(H, hd)[:, :half] + k_r.reshape(H, hd)[:, half:]) / 2, 0)
        e["q_pair__k_pair"] = corr2(qp, kp); e["n_pairs"] = int(half)
        # permutation null on the down column profile (Pearson)
        nulls = np.array([np.corrcoef(g_r, RNG.permutation(d_c))[0, 1] for _ in range(N_NULL)])
        e["null_gate_down_p95_abs"] = float(np.quantile(np.abs(nulls), 0.95))
        nulls_v = np.array([np.corrcoef(v_r, RNG.permutation(o_c))[0, 1] for _ in range(N_NULL)])
        e["null_v_o_p95_abs"] = float(np.quantile(np.abs(nulls_v), 0.95))
        # mismatch: gate row of layer L vs down column of layer L+1 (wrap to 0 at the last layer)
        Ln = Ls[(Ls.index(L) + 1) % len(Ls)]
        e["mismatch_gate_row__down_col_next_layer"] = corr2(g_r, prof(lay[Ln]["down_proj"], 0))
        e["mismatch_v_row__o_col_next_layer"] = corr2(v_r, prof(lay[Ln]["o_proj"], 0))
        e["H_gate_row"] = float(np.log(np.sqrt((m["gate_proj"] ** 2).mean(1))).std())
        e["H_down_col"] = float(np.log(np.sqrt((m["down_proj"] ** 2).mean(0))).std())
        out.append(e)
        if fam == "tclock" and run == "D1_s1" and step == 5000 and L == 3:
            e["_example"] = {"gate_row": g_r.tolist(), "down_col": d_c.tolist()}
    return out


def run_meta(fam, run):
    if fam == "tclock":
        a, s = run.split("_s"); p = ROOT / f"04_tclock_runs/wt_ckpts/tclock_{a}_s{s}/run_meta.json"
        if not p.exists(): p = ROOT / f"04_tclock_runs/tclock_{a}_s{s}/run_meta.json"
    elif fam == "ea":
        i, s = run.split("_s"); p = ROOT / f"04_ea_runs/ea_{i}_s{s}/run_meta.json"
    else:
        p = ROOT / "05_ivn_runs/ivn_base/run_meta.json"
    j = json.load(open(p)) if p.exists() else {}
    heads = None
    for k in ("n_heads", "num_attention_heads", "heads", "n_head"):
        if k in j: heads = int(j[k]); break
        if "config" in j and k in j["config"]: heads = int(j["config"][k]); break
        if "model" in j and isinstance(j["model"], dict) and k in j["model"]: heads = int(j["model"][k]); break
    return {"heads": heads, "meta_path": str(p)}


# heads: take from the IVN prereg record (8 heads, head_dim 64) and require every run_meta that
# states a head count to agree
IVN = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
HEADS = int(IVN["heads"]); HEAD_DIM = int(IVN["head_dim"])
metas = {}
for fam, run, step, p in CK:
    if (fam, run) not in metas:
        mt = run_meta(fam, run); metas[(fam, run)] = mt
        if mt["heads"] is not None:
            assert mt["heads"] == HEADS, (fam, run, mt)
META = {"heads": HEADS, "head_dim": HEAD_DIM}

rows = []
for fam, run, step, p in CK:
    rows.extend(analyse_ckpt(fam, run, step, p, META))
    print(fam, run, step, "done", flush=True)
assert len(rows) == len(CK) * 6, len(rows)
example = next(r.pop("_example") for r in rows if "_example" in r)

KEYS = ["gate_row__down_col", "up_row__down_col", "gate_row__up_row", "v_row__o_col", "v_head__o_head",
        "q_row__k_row", "q_pair__k_pair", "mismatch_gate_row__down_col_next_layer", "mismatch_v_row__o_col_next_layer"]


def summ(sel, key, idx=0):
    v = np.array([r[key][idx] for r in sel])
    return {"n": int(v.size), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def block(sel):
    d = {k: {"pearson": summ(sel, k, 0), "spearman": summ(sel, k, 1)} for k in KEYS}
    d["null_gate_down_p95_abs_median"] = float(np.median([r["null_gate_down_p95_abs"] for r in sel]))
    d["null_v_o_p95_abs_median"] = float(np.median([r["null_v_o_p95_abs"] for r in sel]))
    return d


summary = {
    "tclock_step5000": block([r for r in rows if r["family"] == "tclock" and r["step"] == 5000]),
    "tclock_by_step": {str(s): block([r for r in rows if r["family"] == "tclock" and r["step"] == s]) for s in STEPS},
    "tclock_by_arm_step5000": {a: block([r for r in rows if r["family"] == "tclock" and r["step"] == 5000 and r["run"].startswith(a)]) for a in ARMS},
    "ea_step30000_by_init": {i: block([r for r in rows if r["family"] == "ea" and r["step"] == 30000 and r["run"].startswith(i)]) for i in INITS},
    "ea_step0_all_inits": block([r for r in rows if r["family"] == "ea" and r["step"] == 0]),
    "ivn_base_step0": block([r for r in rows if r["family"] == "ivn" and r["step"] == 0]),
    "ivn_base_step30000": block([r for r in rows if r["family"] == "ivn" and r["step"] == 30000]),
    "tclock_step5000_by_layer": {str(L): {k: summ([r for r in rows if r["family"] == "tclock" and r["step"] == 5000 and r["layer"] == L], k) for k in KEYS} for L in range(6)},
}

json.dump({"schema": "p5v2-identity-transfer-v1", "script": Path(__file__).name, "script_sha256": SHA,
           "protocol": "raw W; profile = ln RMS along the axis minus its median; Pearson & Spearman; null = %d shuffles of the column profile; mismatch = next layer (wrap)" % N_NULL,
           "heads": HEADS, "head_dim": HEAD_DIM, "n_ckpt": len(CK), "n_rows": len(rows),
           "summary": summary, "example_D1_s1_step5000_L3": example, "rows": rows},
          open(OUT_JSON, "w"), indent=1)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})
fig = plt.figure(figsize=(13.4, 3.9)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.32)
t5 = [r for r in rows if r["family"] == "tclock" and r["step"] == 5000]
ARMCOL = {"D1": "#1b9e77", "D2": "#7570b3", "D3": "#d95f02", "D4": "#e7298a"}


def layer_panel(ax, key, mkey, nullkey, title, color):
    for r in t5:
        ax.scatter(r["layer"] + RNG.uniform(-0.18, 0.18), r[key][0], s=14, color=ARMCOL[r["run"][:2]], alpha=0.75, zorder=3)
        ax.scatter(r["layer"] + RNG.uniform(-0.18, 0.18), r[mkey][0], s=10, color="0.55", alpha=0.6, marker="x", zorder=2)
    med = [np.median([r[key][0] for r in t5 if r["layer"] == L]) for L in range(6)]
    ax.plot(range(6), med, color=color, lw=2.0, zorder=4)
    p95 = np.median([r[nullkey] for r in t5])
    ax.axhspan(-p95, p95, color="0.85", zorder=1)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xlabel("layer"); ax.set_ylim(-0.25, 1.02)
    ax.set_title(title, loc="left", pad=6)


ax = fig.add_subplot(gs[0, 0])
layer_panel(ax, "gate_row__down_col", "mismatch_gate_row__down_col_next_layer", "null_gate_down_p95_abs",
            "(a) gate rows vs down columns", C_TRA)
ax.set_ylabel("Pearson correlation of profiles")
h = [Line2D([], [], marker="o", ls="", color=ARMCOL[a], label=a) for a in ARMS] + \
    [Line2D([], [], marker="x", ls="", color="0.55", label="mismatch: next-layer down"),
     Line2D([], [], color=C_TRA, lw=2, label="median over 12 runs")]
ax.legend(handles=h, fontsize=7.0, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, handletextpad=0.3, columnspacing=0.8)
fig.text(0.01, -0.02, "(a)-(c): TCLOCK, 12 runs (4 data arms x 3 seeds), step 5,000, per layer; grey band = 95th percentile of |r| under shuffling of the column profile", fontsize=7.6, color="0.3")

ax = fig.add_subplot(gs[0, 1])
layer_panel(ax, "v_row__o_col", "mismatch_v_row__o_col_next_layer", "null_v_o_p95_abs",
            "(b) v rows vs o columns", C_TRA)

ax = fig.add_subplot(gs[0, 2])
for r in t5:
    ax.scatter(r["layer"] - 0.12 + RNG.uniform(-0.08, 0.08), r["q_row__k_row"][0], s=14, color=C_SEL, alpha=0.7, zorder=3)
    ax.scatter(r["layer"] + 0.12 + RNG.uniform(-0.08, 0.08), r["q_pair__k_pair"][0], s=14, color=C_SEL, alpha=0.7, marker="s", facecolors="none", zorder=3)
ax.plot(range(6), [np.median([r["q_row__k_row"][0] for r in t5 if r["layer"] == L]) for L in range(6)], color=C_SEL, lw=2.0)
ax.plot(range(6), [np.median([r["q_pair__k_pair"][0] for r in t5 if r["layer"] == L]) for L in range(6)], color=C_SEL, lw=1.4, ls=(0, (4, 2)))
ax.axhline(0, color="0.4", lw=0.8); ax.set_xticks(range(6)); ax.set_xlabel("layer"); ax.set_ylim(-0.25, 1.0)
ax.set_title("(c) q rows vs k rows", loc="left", pad=6)
h = [Line2D([], [], marker="o", ls="", color=C_SEL, label="per row (%d)" % t5[0]["n_model"]),
     Line2D([], [], marker="s", ls="", mfc="none", color=C_SEL, label="per RoPE pair (%d), median over heads" % t5[0]["n_pairs"])]
ax.legend(handles=h, fontsize=7.2, frameon=False, loc="lower right")

ax = fig.add_subplot(gs[0, 3])
gx = np.array(example["gate_row"]); dy = np.array(example["down_col"])
ax.scatter(gx, dy, s=5, color=C_TRA, alpha=0.45)
rr, rs = corr2(gx, dy)
ax.text(0.03, 0.97, "D1 s1, step 5,000, layer 3\n%d hidden units\nPearson %.2f, Spearman %.2f" % (gx.size, rr, rs), transform=ax.transAxes, va="top", fontsize=8)
ax.set_xlabel("gate row profile  ln RMS − median"); ax.set_ylabel("down column profile  ln RMS − median")
ax.set_title("(d) one layer, per hidden unit", loc="left", pad=6)
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT_FIG, OUT_JSON)
for k in KEYS:
    s = summary["tclock_step5000"][k]["pearson"]; print(f"{k:42s} tclock5000 median {s['median']:+.3f} [{s['min']:+.3f},{s['max']:+.3f}]")
print("null p95 |r| gate-down", summary["tclock_step5000"]["null_gate_down_p95_abs_median"], "v-o", summary["tclock_step5000"]["null_v_o_p95_abs_median"])
for i in INITS:
    print(i, "30k gate-down", summary["ea_step30000_by_init"][i]["gate_row__down_col"]["pearson"], "v-o", summary["ea_step30000_by_init"][i]["v_row__o_col"]["pearson"])
print("step0 EA gate-down", summary["ea_step0_all_inits"]["gate_row__down_col"]["pearson"], "v-o", summary["ea_step0_all_inits"]["v_row__o_col"]["pearson"], "q-k", summary["ea_step0_all_inits"]["q_row__k_row"]["pearson"])
