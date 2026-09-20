#!/usr/bin/env python3
"""v2 (260921): second block with read-outs of the 63-condition set (a=1 cost, a=2/a=1, extreme-decile shares, decile sum, cost/width^2) so the appendix text need not list them. v1: size-matched control edits on two slices. Reads p5v3_gain_edit_matched_controls_v1.json and
p5v3_gain_edit_perturbation_v1.json only."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "data"
MC = json.load(open(R / "p5v3_gain_edit_matched_controls_v1.json")); PT = json.load(open(R / "p5v3_gain_edit_perturbation_v1.json"))
assert len(MC["conditions"]) == 54
P = ("QK", "VO", "UD"); SL = ("slice1", "slice2")
def cs(p, m, **kw): return [c for c in MC["conditions"] if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
FL = {p: cs(p, "flatten1")[0] for p in P}
rows = [("flatten (reference)", "flatten1", {}), ("gain doubled ($a=-1$)", "amplify", {"a": -1.0}), ("gain half-doubled ($a=-0.5$)", "amplify", {"a": -0.5}),
        ("shuffle, size-matched (5 seeds)", "shuffle_matched", {}), ("Gaussian direction (5 seeds)", "gauss", {}), ("random signs (5 seeds)", "signflip", {})]
out = [r"\begin{table}[H]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}l rrr rrr rrr@{}}", r"\toprule",
       r"& \multicolumn{3}{c}{$q/k$} & \multicolumn{3}{c}{$v/o$} & \multicolumn{3}{c}{up/down} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
       r"edit & size & s1 & s2 & size & s1 & s2 & size & s1 & s2 \\", r"\midrule"]
for name, m, kw in rows:
    cells = []
    for p in P:
        c = cs(p, m, **kw); size = np.mean([x["relF2"] for x in c]) / FL[p]["relF2"]
        cells.append(f"{size:.2f}")
        for s in SL: cells.append(f"{np.mean([x['dloss'][s] for x in c]) / FL[p]['dloss'][s]:.2f}")
    out.append(name + " & " + " & ".join(cells) + r" \\")
    if m == "flatten1": out.append(r"\midrule")
# unmatched shuffle from the 63-edit set, for reference
out.append(r"\midrule")
E1 = json.load(open(R / "p5v3_gain_edit_eval_v1.json")); E2 = json.load(open(R / "p5v3_gain_edit_eval_slice2.json"))
cells = []
for p in P:
    rel = np.mean([c["relF2"] for c in PT if c["path"] == p and c["mode"] == "shuffle"]) / FL[p]["relF2"]; cells.append(f"{rel:.2f}")
    for E, s in ((E1, "slice1"), (E2, "slice2")):
        sh = np.mean([c["dloss"] for c in E["conditions"] if c["path"] == p and c["mode"] == "shuffle"]); cells.append(f"{sh / FL[p]['dloss'][s]:.2f}")
out.append("shuffle at the histogram's width (5 seeds) & " + " & ".join(cells) + r" \\")
# ---- second block: read-outs of the 63-condition set (sizes relative to the flatten, from the perturbation file)
out.append(r"\midrule")
out.append(r"\multicolumn{10}{@{}l}{\emph{Read-outs of the 63-condition set (cost relative to the flatten unless stated)}} \\")
out.append(r"\midrule")
def dl(E, p, m, **kw): return [c["dloss"] for c in E["conditions"] if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
def rf(p, m, **kw): return np.mean([c["relF2"] for c in PT if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]) / FL[p]["relF2"]
ES = ((E1, "slice1"), (E2, "slice2"))
def row(name, fsize, fval):
    cells = []
    for p in P:
        cells.append(fsize(p))
        for E, s in ES: cells.append(fval(E, s, p))
    out.append(name + " & " + " & ".join(cells) + r" \\")
f1 = lambda E, p: dl(E, p, "flatten", a=1.0)[0]
row("flatten cost, nats", lambda p: "1.00", lambda E, s, p: f"{f1(E, p):.3f}")
row("gain reversed ($a=2$)", lambda p: f"{rf(p, 'flatten', a=2.0):.2f}", lambda E, s, p: f"{dl(E, p, 'flatten', a=2.0)[0] / f1(E, p):.2f}")
row("most amplified decile alone", lambda p: f"{rf(p, 'decile', decile=9):.2f}", lambda E, s, p: f"{dl(E, p, 'decile', decile=9)[0] / f1(E, p):.2f}")
row("most suppressed decile alone", lambda p: f"{rf(p, 'decile', decile=0):.2f}", lambda E, s, p: f"{dl(E, p, 'decile', decile=0)[0] / f1(E, p):.2f}")
row("sum of the ten deciles", lambda p: "--", lambda E, s, p: f"{sum(dl(E, p, 'decile')) / f1(E, p):.2f}")
row("flatten cost / squared gain width", lambda p: "--", lambda E, s, p: f"{f1(E, p) / E['gain_width'][p] ** 2:.1f}")
print("base loss", E1["base_loss"], E2["base_loss"], "widths", E1["gain_width"])
out += [r"\bottomrule", r"\end{tabular}",
        r"\caption{\textbf{Size-matched control edits and read-outs of the full edit set.} Each entry is the mean change in loss of the edit divided by that of flattening the gain ($a=1$) on the same path and slice (s1, s2); size is the relative squared Frobenius change of the twelve edited matrices, $\|\Delta W\|^2/\|W\|^2$ summed over the path, divided by that of the flatten. The shuffle at the histogram's width is the permutation of Section~\ref{sec:gainedit} without size matching. The lower block reads the 63-condition set of the same two slices: the flatten cost in nats (base loss 3.347 on s1, 3.093 on s2), and, relative to it, the reversal $a=2$, the two extreme deciles of $g$ edited alone, the sum of all ten decile edits, and the flatten cost divided by the squared gain width (0.275, 0.209, 0.068 on the three paths).}",
        r"\label{tab:gainedit_matched}", r"\end{table}"]
Path(__file__).resolve().parents[1].joinpath("../app/tab_gainedit_matched.tex").resolve().write_text("\n".join(out) + "\n")
print("\n".join(out))
