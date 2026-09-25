"""Figure 1: one reading path from the linear map to the two pooled-shape regimes.

Four layers, drawn as one continuous chain:
  (1) y = Wx            -- columns index input channels, rows index output channels
  (2) W = s D_r Z D_c   -- overall scale, row field, column field, normalized core
  (3) three resolutions -- s -> level; field widths -> pooled shape; full arrangement -> identities
  (4) the fields over training -> the two pooled-shape regimes

Layers (1)-(3) are schematic; the matrices in (2) and both panels of (4) are measured.
Zero GPU. Replaces P5v3_F1_schematic.png.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "04_ea_runs/ckpt_key/ea_gaussian_s1/ckpt/step30000.pt"
KEY = "model.layers.3.self_attn.q_proj.weight"
TRAJ = ROOT / "08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v3_F1_overview.png"
OUT_JSON = ROOT / "08_paper5_draft/data/p5v3_fig1_overview_readouts_v1.json"

ROWC, COLC, GRN, ORG = "#c0392b", "#2471a3", "#117a65", "#b9770e"
ROW_ID = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}   # identity on the output rows
COL_ID = {"o_proj", "down_proj"}                                  # identity on the input columns
NICE = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
        "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
ACC, FALL = ("q_proj", "k_proj"), ("gate_proj", "up_proj", "down_proj")


def bothnorm_tracked(w, it=200, tol=1e-6):
    """Verbatim balancing protocol of p5v2_twoaxis_bridge_v1.py."""
    scale = np.sqrt((w ** 2).mean())
    a = np.ones(w.shape[0]); b = np.ones(w.shape[1]); z = w.copy()
    for _ in range(it):
        r = np.sqrt((z ** 2).mean(1)); z = z / r[:, None]; a = a * r
        c = np.sqrt((z ** 2).mean(0)); z = z / c[None, :]; b = b * c
        rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
        if rr.std() / rr.mean() < tol and cc.std() / cc.mean() < tol:
            break
    g = scale / np.sqrt((z ** 2).mean()); z = z * g
    a = a / np.sqrt(g); b = b / np.sqrt(g)
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, "not converged"
    assert np.abs(a[:, None] * z * b[None, :] - w).max() < 1e-9 * scale, \
        "W = diag(a) Z diag(b) violated"
    return z, a, b


# ------------------------------------------------------------------ measured input
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
W = sd[KEY].double().numpy()          # float64: the 1e-9 reconstruction gate is tighter than float32
Z, a, b = bothnorm_tracked(W)
s = float(np.sqrt((W ** 2).mean()))
H_row, H_col = float(np.std(np.log(a))), float(np.std(np.log(b)))

B = 8                                  # the field lives at the scale of a row, not of an entry
ro, co = np.argsort(-a), np.argsort(-b)


def blocks(M):
    M = np.abs(M[ro][:, co])
    n, m = M.shape[0] // B * B, M.shape[1] // B * B
    return M[:n, :m].reshape(n // B, B, m // B, B).mean((1, 3))


Wb, Zb = blocks(W), blocks(Z)
vmax = float(max(Wb.max(), Zb.max())); vmin = float(min(Wb.min(), Zb.min()))
dr = np.log(a)[ro] - np.log(a).mean()
dc = np.log(b)[co] - np.log(b).mean()
flim = float(max(np.abs(dr).max(), np.abs(dc).max()))

traj = json.load(open(TRAJ))
# Aggregation copied from p5v3_fig4_field_evolution_v1.py so the two figures cannot disagree:
# Gaussian family only; both the raw and the identity-axis fit must pass the R^2 gate; the medians
# of k_raw and of k_ident are taken separately over 3 seeds x 6 layers and then subtracted.
GATE = traj["r2_gate"]
steps = sorted({r["step"] for r in traj["per_matrix"]})
H_t, D_t, n_drop = {}, {}, 0
for c in NICE:
    axk = "row" if c in ROW_ID else "col"
    hs, ds = [], []
    for st_ in steps:
        h, kr, ki = [], [], []
        for r in traj["per_matrix"]:
            if r["comp"] != c or r["step"] != st_ or r["family"] != "gaussian":
                continue
            if r["r2_raw"] < GATE or r[f"r2_{axk}"] < GATE:
                n_drop += 1; continue
            h.append(r["H_row"] if axk == "row" else r["H_col"])
            kr.append(r["k_raw"]); ki.append(r[f"k_{axk}"])
        assert h, (c, st_)
        hs.append(float(np.median(h)))
        ds.append(float(np.median(kr)) - float(np.median(ki)))
    H_t[c], D_t[c] = np.array(hs), np.array(ds)

readouts = dict(ckpt=str(CKPT.relative_to(ROOT)), key=KEY, shape=list(W.shape), block=B,
                s=s, H_row=H_row, H_col=H_col,
                recon_max_err=float(np.abs(a[:, None] * Z * b[None, :] - W).max()),
                Z_row_sd=float(np.std(np.log(np.sqrt((Z ** 2).mean(1))))),
                Z_col_sd=float(np.std(np.log(np.sqrt((Z ** 2).mean(0))))),
                traj_source=str(TRAJ.relative_to(ROOT)), steps=steps,
                r2_gate=GATE, n_fits_dropped=n_drop,
                H_end={NICE[c]: float(H_t[c][-1]) for c in NICE},
                depart_end={NICE[c]: float(D_t[c][-1]) for c in NICE})

# ------------------------------------------------------------------ canvas
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(11.6, 12.4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[2.60, 1.0], hspace=0.14, wspace=0.20,
              left=0.055, right=0.975, top=0.985, bottom=0.060)
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.set_aspect("equal")


def head(y, n, txt):
    ax.text(0, y, n, fontsize=12.5, fontweight="bold", va="center", color="0.15")
    ax.text(4.2, y, txt, fontsize=10.5, va="center", color="0.15")


def down(x, y0, y1, col="0.45", lw=1.6):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color=col, shrinkA=0, shrinkB=0))


# ---- (1) the linear map ----------------------------------------------------
head(72, "1", "Each projection is a linear map.  Columns index input channels, rows index output channels.")
ax.add_patch(Rectangle((8, 58), 2.2, 11, fc="0.96", ec=ROWC, lw=1.4))
ax.text(9.1, 70.0, "$y$", ha="center", fontsize=10)
ax.text(12.6, 63.5, "$=$", ha="center", va="center", fontsize=12)
ax.add_patch(Rectangle((15, 58), 11, 11, fc="0.96", ec="0.45", lw=1.0))
ax.text(20.5, 63.5, "$W$", ha="center", va="center", fontsize=13)
ax.plot([15, 15], [58, 69], lw=4, color=ROWC, solid_capstyle="butt")
ax.plot([15, 26], [69, 69], lw=4, color=COLC, solid_capstyle="butt")
ax.text(28.6, 63.5, r"$\times$", ha="center", va="center", fontsize=11)
ax.add_patch(Rectangle((31, 58), 2.2, 11, fc="0.96", ec=COLC, lw=1.4))
ax.text(32.1, 70.0, "$x$", ha="center", fontsize=10)
ax.text(38, 66.0, "rows = output channels", color=ROWC, fontsize=9.5, va="center")
ax.text(38, 61.5, "columns = input channels", color=COLC, fontsize=9.5, va="center")
down(20.5, 57.0, 52.2)

# ---- (2) the decomposition -------------------------------------------------
head(50, "2", "The trained matrix decomposes exactly into a scale, two scale fields and a core.")
kw = dict(cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
fkw = dict(cmap="coolwarm", vmin=-flim, vmax=flim, interpolation="nearest")
axW = ax.inset_axes([8, 30, 12, 12], transform=ax.transData)
axW.imshow(Wb, aspect="auto", **kw); axW.set_xticks([]); axW.set_yticks([])
ax.text(14, 28.4, "$W$", ha="center", va="top", fontsize=11)
ax.text(23.0, 36.6, r"$=\ s\ \cdot$", ha="center", va="center", fontsize=12)
ax.text(23.0, 33.4, rf"${s:.3f}$", ha="center", va="center", fontsize=8, color="0.4")

axR = ax.inset_axes([28.0, 30, 2.2, 12], transform=ax.transData)
axR.imshow(dr[:, None], aspect="auto", **fkw); axR.set_xticks([]); axR.set_yticks([])
ax.text(29.1, 28.4, "$D_r$", ha="center", va="top", fontsize=11, color=ROWC)
axZ = ax.inset_axes([32.6, 30, 12, 12], transform=ax.transData)
axZ.imshow(Zb, aspect="auto", **kw); axZ.set_xticks([]); axZ.set_yticks([])
ax.text(38.6, 28.4, "$Z$", ha="center", va="top", fontsize=11)
axC = ax.inset_axes([32.6, 43.0, 12, 2.2], transform=ax.transData)
axC.imshow(dc[None, :], aspect="auto", **fkw); axC.set_xticks([]); axC.set_yticks([])
ax.text(45.6, 44.1, "$D_c$", ha="left", va="center", fontsize=11, color=COLC)

ax.text(51.0, 40.5, r"$W = s\,D_r\,Z\,D_c$", fontsize=11.5, va="center")
ax.text(51.0, 35.6, "exact, and unique at this gauge up to one constant.\n"
        "Every row and every column of $Z$ has unit RMS.", fontsize=9, va="center", color="0.35")
ax.text(51.0, 31.0, rf"$H^W_{{\mathrm{{row}}}}={H_row:.2f}$,   $H^W_{{\mathrm{{col}}}}={H_col:.2f}$",
        fontsize=9.5, va="center", color="0.35")

# ---- (3) three resolutions -------------------------------------------------
head(24.0, "3", "Three resolutions of the same matrix.  Only the last keeps which channel is which.")
BOX_TOP, BOX_BOT = 17.4, 4.0
for x0, w, col, title, mid, foot in [
    (2.0, 28.0, "0.45", "$s$", r"$\rightarrow\ \lambda$,  RMS", "the overall level"),
    (35.5, 28.0, ORG, r"$H^W_{\mathrm{row}},\,H^W_{\mathrm{col}}$",
     r"$\rightarrow\ k_{\mathrm{raw}}$   via   $\Delta_k^{\mathrm{mix}}=\alpha_P(H^W)^2$",
     "how unequal the channels became"),
    (69.0, 29.0, GRN, "the fields themselves", None, "which channel is which"),
]:
    ax.add_patch(FancyBboxPatch((x0, BOX_BOT), w, BOX_TOP - BOX_BOT,
                                boxstyle="round,pad=0.35,rounding_size=0.8",
                                fc="1.0", ec=col, lw=1.3))
    down(x0 + w / 2, 21.6, BOX_TOP + 0.6, col)
    ax.text(x0 + w / 2, 15.2, title, ha="center", va="center", fontsize=10.5)
    if mid is not None:
        ax.text(x0 + w / 2, 10.8, mid, ha="center", va="center", fontsize=9.5)
    else:   # the three channel identities, one per line
        for i, (pr, lab) in enumerate([(r"$q \leftrightarrow k$", "RoPE frequency pair"),
                                       (r"$v \leftrightarrow o$", "head/value channel"),
                                       (r"up $\leftrightarrow$ down", "FFN hidden unit")]):
            yy = 13.0 - i * 2.4
            ax.text(x0 + 9.6, yy, pr, ha="right", va="center", fontsize=9, color=GRN)
            ax.text(x0 + 11.0, yy, lab, ha="left", va="center", fontsize=8.2, color=GRN)
    ax.text(x0 + w / 2, 5.6, foot, ha="center", va="center", fontsize=8.5, color=col)

ax.annotate("", xy=(97.5, 1.8), xytext=(2.0, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.2, ls=(0, (5, 3))))
ax.text(49.7, 0.3, "increasing resolution;  pooled statistics stop before the last step",
        ha="center", va="bottom", fontsize=8.5, color="0.5")

# ---- (4) the fields over training ------------------------------------------
axH = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])
st = np.array(steps, dtype=float); st[0] = 400.0          # step 0 on a log axis
OFF_H = {"q": (5, 0), "k": (5, 0), "gate": (5, 0), "up": (5, 4), "down": (5, -4)}
OFF_D = {"q": (5, 0), "k": (5, 0), "up": (5, 6), "down": (5, 0), "gate": (5, -6)}
for c in NICE:
    on = c in ACC or c in FALL
    col = ROWC if c in ACC else (ORG if c in FALL else "0.78")
    for axx, series, off in ((axH, H_t[c], OFF_H), (axD, D_t[c], OFF_D)):
        axx.plot(st, series, lw=2.0 if on else 1.2, color=col, alpha=1.0 if on else 0.9,
                 marker="o", ms=3 if on else 0, zorder=3 if on else 1)
        if on:
            axx.annotate(NICE[c], (st[-1], series[-1]), textcoords="offset points",
                         xytext=off[NICE[c]], fontsize=8.5, color=col, va="center")
for axx, lab in ((axH, r"identity-axis field width  $H^W$"),
                 (axD, r"pooled-shape departure  $k_{\mathrm{raw}}-k_{\mathrm{ident}}$")):
    axx.set_xscale("log"); axx.set_xlabel("training step")
    axx.set_ylabel(lab, fontsize=9)
    axx.spines[["top", "right"]].set_visible(False)
    axx.set_xticks([400, 1000, 3200, 10000, 30000])
    axx.set_xticklabels(["0", "1k", "3.2k", "10k", "30k"])
    axx.set_xlim(360, 52000)
axD.axhline(0, lw=0.7, color="0.6")
axH.text(0.03, 0.97, "4   The fields have a history.", transform=axH.transAxes, va="top",
         fontsize=10.5, fontweight="bold", color="0.15")
axH.text(0.03, 0.885, "$q$ and $k$ keep accumulating (red); the FFN fields\n"
         "peak and fall back (orange); $v$ and $o$ are intermediate (grey).",
         transform=axH.transAxes, va="top", fontsize=8.5, color="0.35")
axD.text(0.03, 0.06, "The departure closes where the field falls back\nand does not where it accumulates.",
         transform=axD.transAxes, va="bottom", fontsize=8.5, color="0.35")

fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
OUT_JSON.write_text(json.dumps(readouts, indent=2))
print("wrote", OUT_FIG)
print(json.dumps({k: v for k, v in readouts.items() if k != "steps"}, indent=2)[:900])
