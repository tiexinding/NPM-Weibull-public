#!/usr/bin/env python3
"""Paper#5 v3 Figure 4：尺度场如何随训练演化，以及宏观形状如何跟随（零 GPU，260915）。

主张（骨架 §6）：**关系不变，变的是沿关系走的轨迹**。
  q/k   场持续累积 ⇒ k_raw 持续偏离
  FFN   场见峰回落 ⇒ k_raw 回到其归一化值
  v/o   居中
读数全部取自既有 `data/p5v2_ea_kbi_trajectory_v1.json`（E-A 9 run × 7 时点 × 7 kind × 6 层 = 2,646
拟合，协议 verbatim 自 p5v2_twoaxis_bridge_v1；与 fig1b 逐矩阵 bitwise 交叉核对通过），
**不重算、协议自动一致**。仅用高斯族（本图讨论的是场的演化，不是初始化并入；后者见 Fig 1(c)）。
门控：k_raw 与身份轴 k 两个拟合都必须过 R² gate。
输出 08_paper5_draft/figures/P5v3_F4_field_evolution.png
      08_paper5_draft/data/p5v3_fig4_field_evolution_v1.json
"""
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json"
FIG = ROOT / "08_paper5_draft/figures/P5v3_F4_field_evolution.png"
OUT = ROOT / "08_paper5_draft/data/p5v3_fig4_field_evolution_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
D = json.load(open(SRC))

# ---- 输入闸门 ----
assert D["n_matrices"] == len(D["per_matrix"]) == 2646, D["n_matrices"]
assert D["fig1b_crosscheck_max_abs_dev"] == 0.0, "上游与 fig1b 的逐矩阵交叉核对未通过"
STEPS = D["steps"]; GATE = float(D["r2_gate"]); FAM = "gaussian"
assert FAM in D["inits"] and STEPS[0] == 0 and STEPS[-1] == 30000, (D["inits"], STEPS)
IDENT = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
         "up_proj": "row", "o_proj": "col", "down_proj": "col"}
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
assert set(KINDS) == set(D["families"]), D["families"]
SH = {k: k.replace("_proj", "") for k in KINDS}
C = {"q": "#B2182B", "k": "#D6604D", "v": "#2166AC", "o": "#4393C3",
     "gate": "#1b9e77", "up": "#66C2A5", "down": "#7570b3"}
GROUP = {"q": "accumulate", "k": "accumulate", "v": "partial", "o": "partial",
         "gate": "fall back", "up": "fall back", "down": "fall back"}
LS = {"accumulate": "-", "partial": "--", "fall back": ":"}

rows = {}
for kind in KINDS:
    ax = IDENT[kind]
    H, KR, KI, DM, N, drop = [], [], [], [], [], 0
    for s in STEPS:
        h, kr, ki, dm = [], [], [], []
        for r in D["per_matrix"]:
            if r["comp"] != kind or r["step"] != s or r["family"] != FAM:
                continue
            if r["r2_raw"] < GATE or r[f"r2_{ax}"] < GATE:
                drop += 1; continue
            h.append(r["H_row"] if ax == "row" else r["H_col"])
            kr.append(r["k_raw"]); ki.append(r[f"k_{ax}"])
            dm.append(r["k_raw"] ** -2 - r[f"k_{ax}"] ** -2)
        assert h, (kind, s)
        H.append(float(np.median(h))); KR.append(float(np.median(kr)))
        KI.append(float(np.median(ki))); DM.append(float(np.median(dm))); N.append(len(h))
    h2 = np.array(H) ** 2; dm = np.array(DM)
    alpha = float((h2 * dm).sum() / (h2 * h2).sum())
    r20 = float(1 - ((dm - alpha * h2) ** 2).sum() / (dm ** 2).sum())
    ip = int(np.argmax(H))
    rows[kind] = {"H": H, "k_raw": KR, "k_ident": KI, "dmix": DM, "n": N, "n_dropped": drop,
                  "alpha_through_origin": alpha, "R2_0": r20,
                  "H_peak_step": STEPS[ip], "H_peak": H[ip], "H_final": H[-1],
                  "peak_over_final": H[ip] / H[-1],
                  "k_raw_at_peak": KR[ip], "k_raw_final": KR[-1], "k_ident_final": KI[-1],
                  "group": GROUP[SH[kind]]}
    assert min(N) >= 12, (kind, N)

SIXPI = 6.0 / np.pi ** 2
res = {"schema": "p5v3-fig4-field-evolution-v1", "script_sha256_16": SHA,
       "source": str(SRC.relative_to(ROOT)), "family": FAM, "steps": STEPS, "r2_gate": GATE,
       "note": "medians over 3 seeds x 6 layers of the Gaussian E-A family; H is on each kind's "
               "identity axis; dmix = k_raw^-2 - k_ident^-2; alpha is the through-origin slope of "
               "dmix against H^2 along each kind's own time course",
       "six_over_pi2": SIXPI, "rows": rows}
OUT.write_text(json.dumps(res, indent=1))

plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0})
fig = plt.figure(figsize=(14.6, 5.4))
gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.055, right=0.985, top=0.845, bottom=0.345)
X = [max(s, 300) for s in STEPS]          # step 0 放在 log 轴左端

# ---------------- (a) 场幅值 ----------------
ax = fig.add_subplot(gs[0])
for kind in KINDS:
    sh = SH[kind]; R = rows[kind]
    ax.plot(X, R["H"], LS[R["group"]], marker="o", ms=3.6, lw=1.7, color=C[sh], label=sh)
    ax.plot([X[STEPS.index(R["H_peak_step"])]], [R["H_peak"]], "v", ms=7,
            color=C[sh], mec="k", mew=.5, zorder=5)
ax.set_xscale("log"); ax.set_xlabel("step   (step 0 drawn at the left edge)")
ax.set_ylabel(r"identity-axis field width $H^W$"); ax.set_ylim(0.015, 0.252)
ax.set_title("(a) two regimes: the field accumulates or falls back", loc="left")
ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=7.0, ncol=2, loc="upper left", framealpha=.95)
ax.text(0.98, 0.03, "▼ = peak of each curve", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.0, color="0.35")

# ---------------- (b) 宏观形状的偏离量 ----------------
ax = fig.add_subplot(gs[1])
ax.axhline(0, color="0.35", lw=1.1)
ax.text(X[1], 0.0013, "back on its own identity-axis normalized value", fontsize=6.9, color="0.35")
for kind in KINDS:
    sh = SH[kind]; R = rows[kind]
    dep = [kr - ki for kr, ki in zip(R["k_raw"], R["k_ident"])]
    ax.plot(X, dep, LS[R["group"]], marker="o", ms=3.6, lw=1.7, color=C[sh])
    ax.annotate(sh, (X[-1], dep[-1]), textcoords="offset points", xytext=(5, -2),
                fontsize=7.0, color=C[sh], va="center")
ax.set_xscale("log"); ax.set_xlabel("step")
ax.set_ylabel(r"shape departure  $k_{\mathrm{raw}} - k_{\mathrm{ident}}$")
ax.set_xlim(X[0] * 0.9, X[-1] * 1.55); ax.set_ylim(top=0.0042)
ax.set_title("(b) the shape departs and, where the field falls back, returns", loc="left")
ax.grid(alpha=.25, lw=.5)
h = [Line2D([], [], color="0.35", ls=LS[g], lw=1.7, label=g) for g in ("accumulate", "partial", "fall back")]
ax.legend(handles=h, fontsize=7.0, loc="lower left", framealpha=.95, title="field regime",
          title_fontsize=7.0)

# ---------------- (c) 参数化桥 ----------------
ax = fig.add_subplot(gs[2])
hmax = max(max(rows[k]["H"]) for k in KINDS) ** 2 * 1.06
xs = np.linspace(0, hmax, 50)
ax.plot(xs, SIXPI * xs, color="0.45", lw=1.2, ls="--", zorder=1)
ax.text(hmax * 0.52, SIXPI * hmax * 0.52, r"  $6/\pi^2$", fontsize=7.4, color="0.4", rotation=0)
for kind in KINDS:
    sh = SH[kind]; R = rows[kind]
    ax.plot(np.array(R["H"]) ** 2, R["dmix"], "-", marker="o", ms=3.4, lw=1.5, color=C[sh])
    ax.annotate("", xy=(R["H"][-1] ** 2, R["dmix"][-1]),
                xytext=(R["H"][-2] ** 2, R["dmix"][-2]),
                arrowprops=dict(arrowstyle="-|>", color=C[sh], lw=1.4))
ax.set_xlabel(r"$(H^W)^2$", labelpad=1)
ax.text(0.5, -0.185, "each curve runs from step 0 to 30,000", transform=ax.transAxes,
        ha="center", fontsize=7.4, color="0.35")
ax.set_ylabel(r"$\Delta_k^{\mathrm{mix}} = k_{\mathrm{raw}}^{-2} - k_{\mathrm{ident}}^{-2}$")
ax.set_title("(c) the same relation governs both regimes;\n"
             "     only the path along it differs", loc="left")
ax.grid(alpha=.25, lw=.5)
amin = min(rows[k]["alpha_through_origin"] for k in KINDS)
amax = max(rows[k]["alpha_through_origin"] for k in KINDS)
rmin = min(rows[k]["R2_0"] for k in KINDS)
ax.text(0.97, 0.06, f"through-origin slope along time:\n"
                    f"$\\alpha$ = {amin:.2f}–{amax:.2f},  $R^2_0 \\geq$ {rmin:.2f}  (7 kinds)",
        transform=ax.transAxes, ha="right", fontsize=7.2, color="0.25")

NOTE = (
    r"Medians over the Gaussian E-A family (3 seeds $\times$ 6 layers) at each checkpoint; "
    r"$H^W$ and $k_{\mathrm{ident}}$ are taken on each kind's identity axis (rows for q, k, v, gate, up; "
    r"columns for o, down)."
    "\n"
    r"Fits are read from the existing two-axis trajectory read-out (2,646 fits, protocol verbatim from the "
    r"two-axis bridge script, cross-checked per matrix against the Figure 1(b) read-outs to 0.0); a matrix "
    r"enters only if both its raw and its"
    "\n"
    r"identity-axis fit pass the $R^2 \geq 0.99$ gate. Line style marks the regime read off panel (a), not a "
    r"fitted grouping. $6/\pi^2$ is the exact-Weibull coefficient, drawn for reference only."
)
fig.text(0.012, 0.255, NOTE, va="top", fontsize=6.9, color="0.3", linespacing=1.55)
fig.text(0.012, 0.015, f"p5v3_fig4_field_evolution_v1.py  sha {SHA}   ·   llama-70M, WikiText-103, "
                       f"Gaussian initialization, 3 seeds", fontsize=6.6, color="0.5")
fig.savefig(FIG, dpi=200, facecolor="white")
print("wrote", FIG.relative_to(ROOT), "and", OUT.relative_to(ROOT))
print(f"\n{'kind':<6s}{'group':<11s}{'H峰步':>7s}{'峰/终':>7s}{'k_raw终':>9s}{'k_id终':>8s}{'α':>7s}{'R²₀':>7s}{'丢弃':>6s}")
for kind in KINDS:
    R = rows[kind]
    print(f"{SH[kind]:<6s}{R['group']:<11s}{R['H_peak_step']:>7d}{R['peak_over_final']:>7.2f}"
          f"{R['k_raw_final']:>9.3f}{R['k_ident_final']:>8.3f}"
          f"{R['alpha_through_origin']:>7.3f}{R['R2_0']:>7.3f}{R['n_dropped']:>6d}")
