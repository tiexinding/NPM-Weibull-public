#!/usr/bin/env python3
"""Appendix Table 5 (tab:sides), generated from p5v2_twoaxis_bridge_v1.json only.
Per kind: medians over the 288 grid fits of k_raw / k_row / k_col / k_bi / H_row / H_col, the through-origin
R0^2 of  k_raw^-2 - k_bi^-2  on (H_row)^2 alone (= stored M1), on (H_col)^2 alone (recomputed with the same
estimator), and on both (= stored M2); the 'identity side' column is the axis assignment of Table 3 in the main text
(rows for q, k, gate, up; columns for o, down; both for v), reproduced here so the reader can compare it with the
three R0^2 values."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "data"
D = json.load(open(R / "p5v2_twoaxis_bridge_v1.json")); pm = D["per_matrix"]; assert len(pm) == 2016
SIDE = {"q_proj": "rows", "k_proj": "rows", "v_proj": "both", "o_proj": "columns", "gate_proj": "rows", "up_proj": "rows", "down_proj": "columns"}
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB = {"q_proj": "$q$", "k_proj": "$k$", "v_proj": "$v$", "o_proj": "$o$", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
def r0(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
rows = []
for k in KINDS:
    m = [r for r in pm if r["kind"] == k]; assert len(m) == 288, (k, len(m))
    med = {q: float(np.median([r[q] for r in m])) for q in ("k_raw", "k_row", "k_col", "k_bi", "H_row", "H_col")}
    y = np.array([r["k_raw"] ** -2 - r["k_bi"] ** -2 for r in m]); hr = np.array([r["H_row"] ** 2 for r in m]); hc = np.array([r["H_col"] ** 2 for r in m])
    R_row, R_col, R_both = r0(hr[:, None], y), r0(hc[:, None], y), r0(np.stack([hr, hc], 1), y)
    S = D["fits"][k]; assert abs(R_row - S["M1"]["R0_sq"]) < 2e-3 and abs(R_both - S["M2"]["R0_sq"]) < 2e-3, (k, R_row, S["M1"]["R0_sq"], R_both, S["M2"]["R0_sq"])
    side = SIDE[k]
    rows.append((LAB[k], med, R_row, R_col, R_both, side))
    print(f"{k:10s} k {med['k_raw']:.3f} {med['k_row']:.3f} {med['k_col']:.3f} {med['k_bi']:.3f}  H {med['H_row']:.2f}/{med['H_col']:.2f}  R0 {R_row:.3f}/{R_col:.3f}/{R_both:.3f}  {side}")
out = [r"\begin{table}[tbp]", r"\centering\small", r"\begin{tabular}{@{}l cccc c c l@{}}", r"\toprule",
       r"kind & $\kraw$ & $k_{\mathrm{row}}$ & $k_{\mathrm{col}}$ & $\kbi$ & $\Hrow$/$\Hcol$ & $R_0^2$ row/col/both & identity side (Table~\ref{tab:paths}) \\", r"\midrule"]
for lab, med, a, b, c, side in rows:
    out.append(f"{lab} & {med['k_raw']:.3f} & {med['k_row']:.3f} & {med['k_col']:.3f} & {med['k_bi']:.3f} & {med['H_row']:.2f} / {med['H_col']:.2f} & {a:.3f} / {b:.3f} / {c:.3f} & {side} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
Path(__file__).resolve().parents[1].joinpath("../app/tab_sides_body.tex").resolve().write_text("\n".join(out) + "\n")
print("wrote tab_sides_body.tex")
