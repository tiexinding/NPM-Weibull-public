"""Paper#5 v2 Figure S11: timing chain over training, three initialization distributions x seven component kinds.

Rows: (1) two-axis core k_bi(t); (2) identity-axis scale heterogeneity H(t) (rows for q/k/gate/up,
columns for o/down, both for v); (3) pooled k_raw(t). Columns: seven component kinds. Lines = family
medians over 3 seeds x 6 layers per checkpoint, three initialization distributions.

Reads only:   08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json  (per_matrix)
              08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json      (per_matrix, fits)  -> G2 table
Writes:       08_paper5_draft/figures/P5v2_S11_timing_chain.png
              08_paper5_draft/data/p5v2_S11_timing_readouts_v1.json
              08_paper5_draft/data/p5v2_normalization_side_table_v1.md
"""
import hashlib, json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "08_paper5_draft/data"
EA = json.load(open(D / "p5v2_ea_kbi_trajectory_v1.json"))
TB = json.load(open(D / "p5v2_twoaxis_bridge_v1.json"))
OUT = ROOT / "08_paper5_draft/figures/P5v2_S11_timing_chain.png"
READ = D / "p5v2_S11_timing_readouts_v1.json"
TABLE = D / "p5v2_normalization_side_table_v1.md"
SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]

COMPS = list(EA["families"]); INITS = list(EA["inits"]); SEEDS = list(EA["seeds"]); STEPS = sorted(int(s) for s in EA["steps"])
K0 = float(EA["k0_gaussian_reference"]); TOL = float(EA["merge_tol"])
rows = EA["per_matrix"]
assert len(rows) == EA["n_matrices"] == len(INITS) * len(SEEDS) * len(STEPS) * len(COMPS) * 6, len(rows)
assert sorted(set(r["step"] for r in rows)) == STEPS
LABEL = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
IDAXIS = {"q_proj": ["H_row"], "k_proj": ["H_row"], "gate_proj": ["H_row"], "up_proj": ["H_row"],
          "o_proj": ["H_col"], "down_proj": ["H_col"], "v_proj": ["H_row", "H_col"]}
C_INIT = {"gaussian": "0.25", "laplace": "#B2182B", "uniform": "#2166AC"}

def med(init, comp, step, q):
    v = [r[q] for r in rows if r["family"] == init and r["comp"] == comp and r["step"] == step]
    assert len(v) == len(SEEDS) * 6, (init, comp, step, len(v))
    return float(st.median(v))

traj = {f"{i}|{c}|{q}": [med(i, c, s, q) for s in STEPS] for i in INITS for c in COMPS for q in ("k_bi", "k_raw", "H_row", "H_col")}

def _sx(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))

def half_reach_step(steps, series):
    """first step at which the series reaches half of its terminal value (linear interpolation between checkpoints)."""
    target = 0.5 * series[-1]
    if series[0] >= target:
        return steps[0]
    for a, b, ya, yb in zip(steps[:-1], steps[1:], series[:-1], series[1:]):
        if ya < target <= yb:
            return float(a + (b - a) * (target - ya) / (yb - ya))
    return None

def first_departure_step(steps, series, ref, tol):
    for s, y in zip(steps, series):
        if abs(y - ref) > tol:
            return s
    return None

read = {"script": Path(__file__).name, "script_sha256": SHA, "steps": STEPS, "k0_reference": K0, "merge_tol": TOL,
        "identity_axis": IDAXIS, "family_median_trajectories": traj, "timing": {}}
for c in COMPS:
    for i in INITS:
        e = {"k_bi_merge_step": None if i == "gaussian" else EA["merge_steps_k_bi"][f"{i}|{c}|k_bi_first_step_within_{TOL}_of_gaussian"],
             "k_bi_gaussian_max_abs_dev_from_k0": max(abs(x - K0) for x in traj[f"gaussian|{c}|k_bi"]) if i == "gaussian" else None}
        for ax in IDAXIS[c]:
            ser = traj[f"{i}|{c}|{ax}"]
            e[f"{ax}_step0"] = ser[0]; e[f"{ax}_30k"] = ser[-1]
            e[f"{ax}_half_of_30k_reached_at_step"] = half_reach_step(STEPS, ser)
        kr = traj[f"{i}|{c}|k_raw"]
        e["k_raw_30k"] = kr[-1]
        # departure measured after the core has merged (for non-Gaussian inits the early transient is the init shape, not a field effect)
        start = 0 if i == "gaussian" else e["k_bi_merge_step"]
        post = [(s, y) for s, y in zip(STEPS, kr) if s >= start]
        e["k_raw_first_step_beyond_tol_after_core_merge"] = next((s for s, y in post if abs(y - K0) > TOL), None)
        e["k_raw_max_abs_dev_after_core_merge"] = max(abs(y - K0) for _, y in post)
        read["timing"][f"{i}|{c}"] = e

# ---------------- figure ----------------
plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 9.5})
fig, axes = plt.subplots(3, len(COMPS), figsize=(3.0 * len(COMPS), 8.4), sharex=True)
xt = [0, 800, 3200, 10000, 30000]; xl = ["0", "800", "3.2k", "10k", "30k"]
for j, c in enumerate(COMPS):
    for i in INITS:
        col = C_INIT[i]
        axes[0, j].plot(_sx(STEPS), traj[f"{i}|{c}|k_bi"], color=col, lw=1.9, marker="o", ms=3, mec="white", mew=0.5)
        for ax_name, ls in (("H_row", "-"), ("H_col", (0, (3, 2)))):
            if ax_name in IDAXIS[c]:
                axes[1, j].plot(_sx(STEPS), traj[f"{i}|{c}|{ax_name}"], color=col, lw=1.9, ls=ls, marker="o", ms=3, mec="white", mew=0.5)
        axes[2, j].plot(_sx(STEPS), traj[f"{i}|{c}|k_raw"], color=col, lw=1.9, marker="o", ms=3, mec="white", mew=0.5)
    axes[0, j].axhline(K0, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=1)
    axes[2, j].axhline(K0, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=1)
    axes[0, j].set_title(f"{LABEL[c]}  ({'rows' if IDAXIS[c]==['H_row'] else 'columns' if IDAXIS[c]==['H_col'] else 'rows + columns'})", loc="left")
    axes[2, j].set_xticks(_sx(xt)); axes[2, j].set_xticklabels(xl, fontsize=8)
    axes[2, j].set_xlabel("training step")
    for r in range(3):
        axes[r, j].spines[["top", "right"]].set_visible(False)
kb_all = [v for k, v in traj.items() if k.endswith("|k_bi")]; kr_all = [v for k, v in traj.items() if k.endswith("|k_raw")]
lo = min(min(v) for v in kb_all + kr_all) - 0.02; hi = max(max(v) for v in kb_all + kr_all) + 0.02
for j in range(len(COMPS)):
    axes[0, j].set_ylim(lo, hi); axes[2, j].set_ylim(lo, hi)
axes[0, 0].set_ylabel(r"two-axis core  $k_{\mathrm{bi}}$")
axes[1, 0].set_ylabel(r"identity-axis spread  $H^W$")
axes[2, 0].set_ylabel(r"pooled  $k_{\mathrm{raw}}$")
h = [Line2D([], [], color=C_INIT[i], lw=1.9, label=f"{i} init") for i in INITS] + \
    [Line2D([], [], color="0.5", lw=1.9, label=r"$H_{\mathrm{row}}$ (solid)"), Line2D([], [], color="0.5", lw=1.9, ls=(0, (3, 2)), label=r"$H_{\mathrm{col}}$ (dashed, $v$ only)")]
axes[1, 0].legend(handles=h, loc="upper left", fontsize=7.6, frameon=False, handlelength=1.8)
fig.text(0.005, 0.005, "E-A runs: llama-70M, one data setting, 3 initialization distributions x 3 seeds, 30k steps; lines = family median over 3 seeds x 6 layers; x linear to 800, log above",
         fontsize=8, color="0.3")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
json.dump(read, open(READ, "w"), indent=1)

# ---------------- G2: normalization-side table from the TCLOCK two-axis json ----------------
pm = TB["per_matrix"]
assert len(pm) == TB["n_matrices"] == 48 * 42, len(pm)
kinds = list(TB["families"])
def fit_origin(X, y):
    X = np.asarray(X); y = np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta; r0 = 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()
    return beta, r0
def loro(sub, cols):
    groups = sorted(set((r["arm"], r["seed"]) for r in sub)); errs = []
    for g in groups:
        tr = [r for r in sub if (r["arm"], r["seed"]) != g]; te = [r for r in sub if (r["arm"], r["seed"]) == g]
        beta, _ = fit_origin([[r[c] ** 2 for c in cols] for r in tr], [r["k_raw"] ** -2 - r["k_bi"] ** -2 for r in tr])
        pred = [(r["k_bi"] ** -2 + sum(b * r[c] ** 2 for b, c in zip(beta, cols))) ** -0.5 for r in te]
        errs.append(float(np.mean([abs(p - r["k_raw"]) for p, r in zip(pred, te)])))
    return float(np.median(errs))
lines = ["# Normalization-side comparison, TCLOCK grid (2,016 matrices = 12 runs x 4 checkpoints x 7 component kinds x 6 layers)", "",
         f"Source `p5v2_twoaxis_bridge_v1.json`; script `{Path(__file__).name}` (sha {SHA}). Medians over all 288 fits per kind; D3/5000 = median over the 3 seeds x 6 layers of the shuffled high-repetition arm at 5,000 steps. Bridge target is $k_{{\\mathrm{{raw}}}}^{{-2}} - k_{{\\mathrm{{bi}}}}^{{-2}}$, fits through the origin; LORO = leave-one-run-out median MAE of predicted $k_{{\\mathrm{{raw}}}}$ over 12 folds.", "",
         "| kind | $k_{\\mathrm{raw}}$ | $k_{\\mathrm{row}}$ | $k_{\\mathrm{col}}$ | $k_{\\mathrm{bi}}$ | D3/5000: raw / row / col / bi | $H_{\\mathrm{row}}$ / $H_{\\mathrm{col}}$ | row-only $R_0^2$ / LORO | col-only $R_0^2$ / LORO | both $R_0^2$ / LORO | side explaining more |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
tab = {}
for kd in kinds:
    sub = [r for r in pm if r["kind"] == kd]; assert len(sub) == 288
    d3 = [r for r in sub if r["arm"] == "D3" and r["step"] == 5000]; assert len(d3) == 18
    m = {q: float(st.median(r[q] for r in sub)) for q in ("k_raw", "k_row", "k_col", "k_bi", "H_row", "H_col")}
    e = {q: float(st.median(r[q] for r in d3)) for q in ("k_raw", "k_row", "k_col", "k_bi")}
    y = [r["k_raw"] ** -2 - r["k_bi"] ** -2 for r in sub]
    res = {}
    for name, cols in (("row", ["H_row"]), ("col", ["H_col"]), ("both", ["H_row", "H_col"])):
        beta, r0 = fit_origin([[r[c] ** 2 for c in cols] for r in sub], y)
        res[name] = {"coef": [float(b) for b in beta], "R0_sq": float(r0), "loro_median": loro(sub, cols)}
    # cross-check against the stored M1 / M2 fits
    assert abs(res["row"]["R0_sq"] - TB["fits"][kd]["M1"]["R0_sq"]) < 1e-6 and abs(res["both"]["R0_sq"] - TB["fits"][kd]["M2"]["R0_sq"]) < 1e-6, kd
    side = "rows" if res["row"]["R0_sq"] > res["col"]["R0_sq"] else "columns"
    tab[kd] = {"medians": m, "D3_5000": e, "fits": res, "side_explaining_more": side}
    lines.append(f"| {LABEL[kd]} | {m['k_raw']:.3f} | {m['k_row']:.3f} | {m['k_col']:.3f} | {m['k_bi']:.3f} | {e['k_raw']:.3f} / {e['k_row']:.3f} / {e['k_col']:.3f} / {e['k_bi']:.3f} | {m['H_row']:.2f} / {m['H_col']:.2f} | {res['row']['R0_sq']:.3f} / {res['row']['loro_median']:.4f} | {res['col']['R0_sq']:.3f} / {res['col']['loro_median']:.4f} | {res['both']['R0_sq']:.3f} / {res['both']['loro_median']:.4f} | {side} |")
lines += ["", "**Caption.** For each component kind the pooled shape $k_{\\mathrm{raw}}$ is refitted after removing the row scale field ($k_{\\mathrm{row}}$), the column field ($k_{\\mathrm{col}}$) or both ($k_{\\mathrm{bi}}$), at fixed global RMS. The side whose removal returns the shape to the reference, and whose heterogeneity explains more of $k_{\\mathrm{raw}}^{-2}-k_{\\mathrm{bi}}^{-2}$, is the row side for $q$, $k$, gate and up and the column side for $o$ and down; for $v$ the two sides are comparable and only the two-sided fit reaches the held-out error of the other kinds. Row-only and both-sides fits reproduce the stored M1/M2 fits of `p5v2_twoaxis_bridge_v1.json` to $10^{-6}$; the column-only fit is computed here."]
TABLE.write_text("\n".join(lines) + "\n")
read["normalization_side_table"] = tab
json.dump(read, open(READ, "w"), indent=1)
print("wrote", READ, TABLE)
for k, e in read["timing"].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in e.items()})
