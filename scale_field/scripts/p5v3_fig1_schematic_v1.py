"""Figure 1 (schematic): the decomposition, the identity axes and paths, and the information ladder.

Panel (a) uses one real trained matrix (no synthetic data); panels (b) and (c) are schematics.
Zero GPU. Writes figures/P5v3_F1_schematic.png and data/p5v3_fig1_schematic_readouts_v1.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "04_ea_runs/ckpt_key/ea_gaussian_s1/ckpt/step30000.pt"
KEY = "model.layers.3.self_attn.q_proj.weight"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v3_F1_schematic.png"
OUT_JSON = ROOT / "08_paper5_draft/data/p5v3_fig1_schematic_readouts_v1.json"


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


# ---------------------------------------------------------------- data (real)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
W = sd[KEY].double().numpy()   # float64: the 1e-9 reconstruction gate is tighter than float32
assert W.ndim == 2 and W.shape[0] >= 64, W.shape
Z, a, b = bothnorm_tracked(W)
s = float(np.sqrt((W ** 2).mean()))
H_row = float(np.std(np.log(a)))
H_col = float(np.std(np.log(b)))
# The field lives at the scale of a whole row or column, not of a single entry, so the maps show
# magnitude averaged over B x B blocks; rows and columns are sorted by their RMS in W and the SAME
# ordering is applied to Z, so any residual field would show as a gradient.
B = 8
ro = np.argsort(-np.sqrt((W ** 2).mean(1)))
co = np.argsort(-np.sqrt((W ** 2).mean(0)))


def blocks(M):
    M = np.abs(M[ro][:, co])
    n, m = M.shape[0] // B * B, M.shape[1] // B * B
    return M[:n, :m].reshape(n // B, B, m // B, B).mean((1, 3))


Wb, Zb = blocks(W), blocks(Z)
vmax = float(max(Wb.max(), Zb.max()))
vmin = float(min(Wb.min(), Zb.min()))
# measured field profiles, sorted the same way, for W and for Z on one common scale
prof = {"W_row": np.log(np.sqrt((W ** 2).mean(1)))[ro], "Z_row": np.log(np.sqrt((Z ** 2).mean(1)))[ro],
        "W_col": np.log(np.sqrt((W ** 2).mean(0)))[co], "Z_col": np.log(np.sqrt((Z ** 2).mean(0)))[co]}
for k_ in prof:
    prof[k_] = prof[k_] - prof[k_].mean()
PLIM = 1.15 * max(np.abs(v).max() for v in prof.values())

readouts = dict(ckpt=str(CKPT.relative_to(ROOT)), key=KEY, shape=list(W.shape),
                s=s, H_row=H_row, H_col=H_col, block=B, Z_row_sd=float(prof["Z_row"].std()), Z_col_sd=float(prof["Z_col"].std()),
                recon_max_err=float(np.abs(a[:, None] * Z * b[None, :] - W).max()))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(13.0, 7.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.50], width_ratios=[1.05, 1.0],
              hspace=0.10, wspace=0.16,
              left=0.045, right=0.985, top=0.94, bottom=0.04)

ROWC, COLC = "#c0392b", "#2471a3"

# ---- (a) the decomposition, on one real trained matrix ----------------------
gsa = GridSpecFromSubplotSpec(3, 5, subplot_spec=gs[0, 0],
                              width_ratios=[0.20, 1.0, 0.34, 0.20, 1.0],
                              height_ratios=[0.20, 1.0, 0.44], hspace=0.05, wspace=0.05)
axWc = fig.add_subplot(gsa[0, 1]); axWr = fig.add_subplot(gsa[1, 0])
axZc = fig.add_subplot(gsa[0, 4]); axZr = fig.add_subplot(gsa[1, 3])
axW = fig.add_subplot(gsa[1, 1]); axZ = fig.add_subplot(gsa[1, 4])
axmid = fig.add_subplot(gsa[1, 2]); axmid.axis("off")

kw = dict(cmap="magma", aspect="equal", vmin=vmin, vmax=vmax, interpolation="nearest")
axW.imshow(Wb, **kw); axZ.imshow(Zb, **kw)
for ax in (axW, axZ):
    ax.set_xticks([]); ax.set_yticks([])
axW.set_xlabel(r"$|W|$   trained $q$, layer 3", fontsize=9, labelpad=3)
axZ.set_xlabel(r"$|Z|$   doubly balanced core", fontsize=9, labelpad=3)

n = len(prof["W_row"]); m = len(prof["W_col"])
for ax, key, col, horiz in ((axWr, "W_row", ROWC, False), (axZr, "Z_row", "0.6", False),
                            (axWc, "W_col", COLC, True), (axZc, "Z_col", "0.6", True)):
    y = prof[key]
    if horiz:
        ax.plot(np.arange(len(y)), y, lw=0.9, color=col)
        ax.axhline(0, lw=0.5, color="0.7")
        ax.set_xlim(-0.5, len(y) - 0.5); ax.set_ylim(-PLIM, PLIM)
    else:
        ax.plot(y, np.arange(len(y)), lw=0.9, color=col)
        ax.axvline(0, lw=0.5, color="0.7")
        ax.set_ylim(len(y) - 0.5, -0.5); ax.set_xlim(-PLIM, PLIM)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

axWr.text(-0.05, 0.5, rf"$h^{{W,r}}$,  $H^W_{{\mathrm{{row}}}}={H_row:.2f}$", transform=axWr.transAxes,
          ha="right", va="center", fontsize=8, color=ROWC, rotation=90)

axWc.text(0.0, 1.10, rf"$h^{{W,c}}$,  $H^W_{{\mathrm{{col}}}}={H_col:.2f}$", transform=axWc.transAxes,
          ha="left", va="bottom", fontsize=8, color=COLC)
axZc.text(0.0, 1.10, "both fields removed:\nrow and column RMS flat to $<10^{-7}$",
          transform=axZc.transAxes, ha="left", va="bottom", fontsize=8, color="0.35")

axmid.text(0.5, 0.55, r"$=\;s\;\cdot$", ha="center", va="center", fontsize=14)
axmid.text(0.5, 0.42, rf"$s={s:.3f}$", ha="center", va="center", fontsize=7.5, color="0.35")

axatxt = fig.add_subplot(gsa[2, 0:5]); axatxt.axis("off")
axatxt.set_xlim(0, 1); axatxt.set_ylim(0, 1)
axatxt.text(0.0, 0.98,
            r"$W = s\,D_r\,Z\,D_c$ — exact, and unique at this gauge up to one constant."
            "\nMaps show magnitude averaged over $8\\times8$ blocks of the full $512\\times512$ matrix, with"
            "\nrows and columns sorted by their RMS in $W$ and the same ordering applied to $Z$. The curves"
            "\nare the measured log row and column RMS on one common scale: in $W$ they vary, in $Z$ they"
            "\nare flat by construction. The fields are what training writes; the core is what is left.",
            fontsize=8.5, va="top", ha="left", color="0.25")

fig.text(0.005, 0.955, "(a)", fontsize=12, fontweight="bold")

# ---- (b) identity axes and paths -------------------------------------------
axb = fig.add_subplot(gs[0, 1]); axb.axis("off")
axb.set_xlim(0, 10); axb.set_ylim(0, 10)
fig.text(0.525, 0.955, "(b)", fontsize=12, fontweight="bold")

BW, BH = 1.55, 1.7
boxes = {
    "$W_q$": (0.55, 7.0, "rows"), "$W_k$": (2.55, 7.0, "rows"),
    "$W_v$": (5.30, 7.0, "both"), "$W_o$": (7.75, 7.0, "cols"),
    "gate": (0.55, 3.3, "rows"), "up": (2.55, 3.3, "rows"), "down": (5.30, 3.3, "cols"),
}
for name, (x, y, side) in boxes.items():
    axb.add_patch(Rectangle((x, y), BW, BH, fc="0.95", ec="0.45", lw=0.9))
    axb.text(x + BW / 2, y + BH / 2, name, ha="center", va="center", fontsize=10.5)
    if side in ("rows", "both"):     # identity on the output rows: left edge
        axb.plot([x, x], [y, y + BH], lw=3.6, color=ROWC, solid_capstyle="butt")
    if side in ("cols", "both"):     # identity on the input columns: top edge
        axb.plot([x, x + BW], [y + BH, y + BH], lw=3.6, color=COLC, solid_capstyle="butt")

GRN = "#117a65"
def link(x0, y0, x1, y1, rad, label, lx, ly, va):
    axb.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", lw=1.5, color=GRN,
                                  linestyle=(0, (4, 2)), connectionstyle=f"arc3,rad={rad}"))
    axb.text(lx, ly, label, ha="center", va=va, fontsize=8, color=GRN)

# q rows <-> k rows (both on the left edge of their box)
link(0.55, 6.95, 2.55, 6.95, 0.45, "RoPE frequency pair\n" r"rows of $q$ $\leftrightarrow$ rows of $k$",
     1.55, 6.05, "top")
# v rows <-> o columns
link(5.30, 7.85, 8.53, 8.70, -0.38, "head/value channel\n" r"rows of $v$ $\leftrightarrow$ columns of $o$",
     6.95, 6.45, "top")
# up rows <-> down columns
link(2.55, 4.15, 6.08, 5.00, -0.38, "FFN hidden unit\n" r"rows of up $\leftrightarrow$ columns of down",
     4.30, 2.75, "top")

axb.text(0.0, 1.85,
         "The identity axis is fixed by the architecture:  "
         + r"$\bf{red}$" + " = the output rows carry a channel identity,\n"
         + r"$\bf{blue}$" + " = the input columns carry it;  $v$ carries one on each side.\n"
         "A unit read on two matrices forms a path. Training writes its gain "
         r"$g=h^{(1)}+h^{(2)}$," "\n" r"not the balance $b=h^{(1)}-h^{(2)}$.",
         fontsize=8.5, va="top", color="0.25")

# ---- (c) the information ladder --------------------------------------------
axc = fig.add_subplot(gs[1, :]); axc.axis("off")
axc.set_xlim(0, 10); axc.set_ylim(0, 10)
fig.text(0.005, 0.325, "(c)", fontsize=12, fontweight="bold")

rungs = [
    (0.55, r"$W$", "the trained matrix"),
    (3.0, r"$(s,\;h^{W,r},\;h^{W,c},\;Z)$", "scale, two fields, core"),
    (5.9, r"$(s,\;H^W_{\mathrm{row}},\;H^W_{\mathrm{col}},\;k_{\mathrm{core}})$", "widths and a fitted core"),
    (8.55, r"$(\lambda,\;k_{\mathrm{raw}})$", "the pooled coordinates"),
]
for x, tex, sub in rungs:
    axc.text(x, 6.6, tex, ha="left", va="center", fontsize=11)
    axc.text(x, 5.3, sub, ha="left", va="center", fontsize=8, color="0.4")

arrows = [
    (2.05, 2.85, "exact\nre-parameterization", "keeps everything", "#117a65"),
    (5.05, 5.75, "compression", "drops WHICH channels\nchanged; fits the core", "#b9770e"),
    (8.05, 8.40, "compression", "mixes core and fields", "#b9770e"),
]
for x0, x1, top, bot, col in arrows:
    axc.annotate("", xy=(x1, 6.6), xytext=(x0, 6.6),
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    axc.text((x0 + x1) / 2, 7.4, top, ha="center", va="bottom", fontsize=8, color=col)
    axc.text((x0 + x1) / 2, 3.9, bot, ha="center", va="top", fontsize=8, color=col)

axc.text(0.0, 1.6,
         "The four factors are complementary coordinates of one decomposition, not successive compressions: "
         "together they reconstruct $W$ exactly.\n"
         r"Two matrices of equal $H^W$ can carry entirely different arrangements over channels, so the object of this paper is the field itself; "
         r"$H^W$ and $k$ are summaries of it.",
         fontsize=8.5, va="top", color="0.25")

fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
OUT_JSON.parent.mkdir(exist_ok=True)
OUT_JSON.write_text(json.dumps(readouts, indent=2))
print("wrote", OUT_FIG)
print(json.dumps(readouts, indent=2))
