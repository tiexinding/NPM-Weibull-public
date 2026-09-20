"""Fig 8 v2 (260918): paired gain edits, zero training. Reads only four JSONs under 08_paper5_draft/data:
  p5v3_gain_edit_eval_v1.json (63 edits, slice 1), p5v3_gain_edit_eval_slice2.json (same, slice 2),
  p5v3_gain_edit_matched_controls_v1.json (perturbation-matched controls, both slices, relF2),
  p5v3_gain_edit_perturbation_v1.json (relF2 of the 63 edits).
(a) balance identity against gain flatten;  (b) cost relative to flatten for the perturbation-matched edits;
(c) share of the flatten cost carried by each gain decile.  Slice 1 filled, slice 2 hollow. Prints caption numbers.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve(); DRAFT = HERE.parents[3]; DATA = DRAFT / "data"
OUT = HERE.parents[1] / "figures" / "P5v3_F8_gain_edit_v2.png"
E1 = json.load(open(DATA / "p5v3_gain_edit_eval_v1.json")); E2 = json.load(open(DATA / "p5v3_gain_edit_eval_slice2.json"))
MC = json.load(open(DATA / "p5v3_gain_edit_matched_controls_v1.json")); PT = json.load(open(DATA / "p5v3_gain_edit_perturbation_v1.json"))
assert len(E1["conditions"]) == 63 and len(E2["conditions"]) == 63 and len(MC["conditions"]) == 54 and len(PT) == 63
PATHS = ["QK", "VO", "UD"]; SL = ["slice1", "slice2"]
COL = {"QK": "#B2182B", "VO": "#2166AC", "UD": "#1B7837"}
LBL = {"QK": "$q/k$ path (RoPE pairs)", "VO": "$v/o$ path", "UD": "up/down path"}
def ge(E, p, m, **kw): return [c["dloss"] for c in E["conditions"] if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
def gm(p, m, s, **kw): return [c["dloss"][s] for c in MC["conditions"] if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
def gr(p, m, **kw): return [c["relF2"] for c in MC["conditions"] if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
FL = {p: {"slice1": ge(E1, p, "flatten", a=1.0)[0], "slice2": ge(E2, p, "flatten", a=1.0)[0]} for p in PATHS}
for p in PATHS:  # the reference run of the control script reproduces the a=1 edit exactly
    assert abs(gm(p, "flatten1", "slice1")[0] - FL[p]["slice1"]) < 1e-6
BAL = {p: max(abs(ge(E1, p, "balance")[0]), abs(ge(E2, p, "balance")[0])) for p in PATHS}
assert max(BAL.values()) < 1e-5

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10})
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.2, 4.4), gridspec_kw=dict(width_ratios=[0.8, 1.35, 1.25], wspace=0.3,
                                                                              left=0.05, right=0.985, top=0.9, bottom=0.17))
MK = dict(slice1=dict(mfc=None, mec="white", mew=0.5), slice2=dict(mfc="white", mew=1.3))
def pt(axis, x, y, p, s, marker="o", ms=6, z=4):
    if s == "slice1": axis.plot(x, y, marker, color=COL[p], ms=ms, mec="white", mew=0.5, zorder=z)
    else: axis.plot(x, y, marker, mfc="white", mec=COL[p], ms=ms, mew=1.3, zorder=z)

# (a) balance vs flatten
w = 0.22
for i, p in enumerate(PATHS):
    for j, s in enumerate(SL):
        x = 1 + (i - 1) * w + (j - 0.5) * 0.09
        ax.bar(x, FL[p][s], 0.085, color=COL[p] if s == "slice1" else "white", edgecolor=COL[p], lw=1.2, zorder=3)
ax.text(0, 0.012, r"$|\Delta L|<10^{-5}$" + "\non all paths,\nboth slices", ha="center", va="bottom", fontsize=8.5, color="0.25")
ax.plot([-0.35, 0.35], [0, 0], color="0.25", lw=1.5, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["balance $b$\nremoved", "gain $g$\nflattened"])
ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 0.30); ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ha = [Line2D([], [], ls="none", marker="s", color=COL[p], ms=7, label=LBL[p].replace(" (RoPE pairs)", "")) for p in PATHS]
la = ax.legend(handles=ha, loc="upper left", ncol=1, frameon=False, fontsize=8, handlelength=1.0, bbox_to_anchor=(0.0, 1.0)); ax.add_artist(la)
hb = [Line2D([], [], ls="none", marker="s", color="0.3", ms=7, label="slice 1"),
      Line2D([], [], ls="none", marker="s", mfc="white", mec="0.3", ms=7, mew=1.2, label="slice 2")]
ax.legend(handles=hb, loc="upper right", ncol=1, frameon=False, fontsize=8, handlelength=1.0, bbox_to_anchor=(1.0, 1.0))
ax.set_ylabel(r"$\Delta L$ (nats / token)")
ax.set_title("(a)  Invariant balance and functional gain", loc="left")
ax.grid(axis="y", alpha=0.2, lw=0.5)

# (b) perturbation-matched edits, cost relative to flatten
CATS = [("flatten1", "flatten\n$-g$"), ("shuffle_matched", "permuted\n$(g_\\pi-g)/\\sqrt{2}$"), ("gauss", "Gaussian\n$-\\epsilon$"),
        ("signflip", "random signs\n$-\\sigma g$"), ("amplify", "doubled\n$+g$")]
for k, (m, _) in enumerate(CATS):
    for i, p in enumerate(PATHS):
        for j, s in enumerate(SL):
            kw = {"a": -1.0} if m == "amplify" else {}
            v = (np.array(gm(p, m, s, **kw)) / FL[p][s]) / (np.array(gr(p, m, **kw)) / gr(p, "flatten1")[0])
            x = k + (i - 1) * 0.22 + (j - 0.5) * 0.08
            if len(v) > 1:
                bx.plot([x, x], [v.min(), v.max()], color=COL[p], lw=1.0, zorder=3)
            pt(bx, x, v.mean(), p, s)
bx.axhline(1.0, color="0.5", lw=0.8, ls=(0, (3, 2)), zorder=1)
bx.set_xticks(range(len(CATS))); bx.set_xticklabels([c[1] for c in CATS])
bx.set_ylabel(r"$(\Delta L_{\rm edit}/\Delta L_{\rm flat})\,/\,(D_{\rm edit}/D_{\rm flat})$")
bx.set_ylim(0, 3.9); bx.set_yticks([0, 1, 2, 3])
bx.set_title("(b)  Sensitivity to controlled gain directions", loc="left")
bx.grid(axis="y", alpha=0.2, lw=0.5)
h1 = [Line2D([], [], ls="none", color=COL[p], marker="o", ms=6, label=LBL[p]) for p in PATHS]
l1 = bx.legend(handles=h1, loc="upper left", ncol=3, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0.0, 1.0)); bx.add_artist(l1)
h2 = [Line2D([], [], ls="none", marker="o", color="0.3", ms=6, label="slice 1"),
      Line2D([], [], ls="none", marker="o", mfc="white", mec="0.3", ms=6, mew=1.3, label="slice 2"),
      Line2D([], [], color="0.3", lw=1.0, label="min--max over 5 seeds")]
bx.legend(handles=h2, loc="upper left", ncol=3, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0.0, 0.92))

# (c) decile share of the flatten cost
FLOOR = 1e-3
xs = np.arange(1, 11)
for p in PATHS:
    for s, E in (("slice1", E1), ("slice2", E2)):
        y = np.array([ge(E, p, "decile", decile=d)[0] for d in range(10)]) / FL[p][s]
        yc = np.clip(y, FLOOR, None)
        cx.plot(xs, yc, "-" if s == "slice1" else "--", color=COL[p], lw=1.2, zorder=2, alpha=0.9)
        for d in range(10):
            pt(cx, xs[d], yc[d], p, s, ms=5)
cx.set_yscale("log"); cx.set_ylim(FLOOR * 0.8, 12)
hc = [Line2D([], [], color=COL[p], lw=1.2, marker="o", ms=5, label=LBL[p]) for p in PATHS]
lc = cx.legend(handles=hc, loc="upper left", ncol=3, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.0, 1.0)); cx.add_artist(lc)
hd = [Line2D([], [], color="0.3", lw=1.2, marker="o", ms=5, label="slice 1 (solid, filled)"),
      Line2D([], [], color="0.3", lw=1.2, ls="--", marker="o", mfc="white", mec="0.3", ms=5, mew=1.3, label="slice 2 (dashed, hollow)")]
cx.legend(handles=hd, loc="upper left", ncol=2, frameon=False, fontsize=8, handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.0, 0.92))
cx.set_yticks([1e-3, 1e-2, 1e-1, 1]); cx.set_yticklabels([r"$\leq 10^{-3}$", "0.01", "0.1", "1"])
cx.set_xticks(xs); cx.set_xlabel("decile of $g$  (1 = most suppressed, 10 = most amplified)", fontsize=9.5, ha="right", x=1.0)
cx.set_ylabel(r"$\Delta L_{\rm decile}\,/\,\Delta L_{\rm full}$")
cx.set_title("(c)  Sensitivity across gain deciles", loc="left")
cx.grid(alpha=0.2, lw=0.5, which="major")
fig.savefig(OUT, dpi=220); print("wrote", OUT)

# caption numbers
print("\n== caption numbers (ratio = raw cost ratio; per-unit = divided by displacement ratio) ==")
for p in PATHS:
    print(p, "flatten a=1 dloss s1/s2 %.3f / %.3f  relF2 %.4f" % (FL[p]["slice1"], FL[p]["slice2"], gr(p, "flatten1")[0]))
    for m in ("shuffle_matched", "gauss", "signflip"):
        r = [np.mean(gm(p, m, s)) / FL[p][s] for s in SL]; print(f"   {m:15s} ratio s1/s2 {r[0]:.2f} / {r[1]:.2f}  relF2 {np.mean(gr(p, m)):.4f}")
    r = [gm(p, "amplify", s, a=-1.0)[0] / FL[p][s] for s in SL]; print(f"   amplify a=-1     ratio s1/s2 {r[0]:.2f} / {r[1]:.2f}  relF2 {gr(p, 'amplify', a=-1.0)[0]:.4f}")
    r = [gm(p, "amplify", s, a=-0.5)[0] / FL[p][s] for s in SL]; print(f"   amplify a=-0.5   ratio s1/s2 {r[0]:.2f} / {r[1]:.2f}  relF2 {gr(p, 'amplify', a=-0.5)[0]:.4f}")
    # per-unit curvature along g vs random direction
    pu_f = [FL[p][s] / gr(p, "flatten1")[0] for s in SL]; pu_g = [np.mean([c["dloss"][s] / c["relF2"] for c in MC["conditions"] if c["path"] == p and c["mode"] == "gauss"]) for s in SL]
    print(f"   per-unit flatten/gauss s1/s2 {pu_f[0]/pu_g[0]:.2f} / {pu_f[1]/pu_g[1]:.2f}")
    rel = {c["decile"]: c["relF2"] for c in PT if c["path"] == p and c["mode"] == "decile"}
    for d in (0, 9):
        sh = [ge(E, p, "decile", decile=d)[0] / FL[p][s] for s, E in (("slice1", E1), ("slice2", E2))]
        pu = [ge(E, p, "decile", decile=d)[0] / rel[d] / (FL[p][s] / gr(p, "flatten1")[0]) for s, E in (("slice1", E1), ("slice2", E2))]
        print(f"   decile {d+1}: share s1/s2 {sh[0]:.2f} / {sh[1]:.2f}   per-unit vs flatten line {pu[0]:.2f} / {pu[1]:.2f}")
    ssum = [sum(ge(E, p, "decile", decile=d)[0] for d in range(10)) / FL[p][s] for s, E in (("slice1", E1), ("slice2", E2))]
    print(f"   sum of deciles / flatten s1/s2 {ssum[0]:.2f} / {ssum[1]:.2f}")
