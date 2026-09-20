#!/usr/bin/env python3
"""Appendix table: all 63 gain-edit conditions on two disjoint token slices. Reads only the two eval JSONs."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "data"
d1 = json.load(open(R / "p5v3_gain_edit_eval_v1.json")); d2 = json.load(open(R / "p5v3_gain_edit_eval_slice2.json"))
PT = json.load(open(R / "p5v3_gain_edit_perturbation_v1.json")); assert len(PT) == 63
def rel(p, m, **kw): return [c["relF2"] for c in PT if c["path"] == p and c["mode"] == m and all(c.get(k) == v for k, v in kw.items())]
FR = {p: rel(p, "flatten", a=1.0)[0] for p in ("QK", "VO", "UD")}
def sz(p, m, **kw): return f"{np.mean(rel(p, m, **kw)) / FR[p]:.2f}"
assert len(d1["conditions"]) == 63 and len(d2["conditions"]) == 63
def get(d, p, m, **kw): return [x["dloss"] for x in d["conditions"] if x["path"] == p and x["mode"] == m and all(x.get(k) == v for k, v in kw.items())]
P = ("QK", "VO", "UD")
def f(v): return f"{1e3*v:.2f}" if abs(v) >= 5e-4 else f"{1e3*v:.3f}"
rows = []
rows.append(("balance $b$", [x for p in P for x in (sz(p, "balance"), f"{1e3*abs(get(d1,p,'balance')[0]):.4f}", f"{1e3*abs(get(d2,p,'balance')[0]):.4f}")]))
for a in (0.25, 0.5, 1.0, 1.5, 2.0):
    rows.append((f"flatten $a={a:g}$", [x for p in P for x in (sz(p, "flatten", a=a), f(get(d1,p,'flatten',a=a)[0]), f(get(d2,p,'flatten',a=a)[0]))]))
rows.append(("shuffle, mean of 5", [x for p in P for x in (sz(p, "shuffle"), f(np.mean(get(d1,p,'shuffle'))), f(np.mean(get(d2,p,'shuffle'))))]))
rows.append(("shuffle, min--max", [x for p in P for x in ("", f"{1e3*min(get(d1,p,'shuffle')):.0f}--{1e3*max(get(d1,p,'shuffle')):.0f}", f"{1e3*min(get(d2,p,'shuffle')):.0f}--{1e3*max(get(d2,p,'shuffle')):.0f}")]))
for k in range(10):
    rows.append((f"decile {k+1}" + (" (lowest $g$)" if k == 0 else " (highest $g$)" if k == 9 else ""), [x for p in P for x in (sz(p, "decile", decile=k), f(get(d1,p,'decile',decile=k)[0]), f(get(d2,p,'decile',decile=k)[0]))]))
out = []
out.append(r"\begin{table}[H]"); out.append(r"\centering\footnotesize"); out.append(r"\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}l rrr rrr rrr@{}}"); out.append(r"\toprule")
out.append(r"& \multicolumn{3}{c}{$q/k$} & \multicolumn{3}{c}{$v/o$} & \multicolumn{3}{c}{up/down} \\")
out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
out.append(r"edit & $D/D_{\rm flat}$ & s1 & s2 & $D/D_{\rm flat}$ & s1 & s2 & $D/D_{\rm flat}$ & s1 & s2 \\"); out.append(r"\midrule")
for name, vals in rows:
    out.append(name + " & " + " & ".join(vals) + r" \\")
    if name.startswith("balance") or name.startswith("flatten $a=2") or name.startswith("shuffle, min"): out.append(r"\midrule")
out.append(r"\bottomrule"); out.append(r"\end{tabular}")
b1, b2 = d1["base_loss"], d2["base_loss"]
cap = (r"\caption{\textbf{All 63 checkpoint edits on two disjoint token slices.} Change in mean loss relative to the unedited model on the same tokens, in units of $10^{-3}$ nats; base loss %.3f on slice 1 (offset $9\times10^7$, Figure~\ref{fig:gainedit}) and %.3f on slice 2 (offset $6\times10^7$); 48 sequences of 512 tokens each. The balance row is the absolute value. $D/D_{\rm flat}$ is the squared relative Frobenius displacement of the edit divided by that of the flatten at $a=1$ on the same path. Deciles are of the gain $g$ within each layer, lowest first.}" % (b1, b2))
out.append(cap)
out.append(r"\label{tab:gainedit_full}"); out.append(r"\end{table}")
Path(__file__).resolve().parents[1].joinpath("../app/tab_gainedit_full.tex").resolve().write_text("\n".join(out) + "\n")
# ratio summary printed for the text
for p in P:
    for d, nm in ((d1, "s1"), (d2, "s2")):
        f1 = get(d,p,'flatten',a=1)[0]; s = np.mean(get(d,p,'shuffle')); dd = [get(d,p,'decile',decile=k)[0] for k in range(10)]; w = d1["gain_width"][p]
        print(p, nm, f"shuf/flat {s/f1:.2f}  a2/a1 {get(d,p,'flatten',a=2)[0]/f1:.2f}  top/flat {dd[9]/f1:.2f}  bot/top {dd[0]/dd[9]:.2f}  sumdec/flat {sum(dd)/f1:.2f}  cost/w2 {f1/w**2:.2f}  shuf cv {np.std(get(d,p,'shuffle'))/s:.3f}")
print("wrote tab_gainedit_full.tex")
