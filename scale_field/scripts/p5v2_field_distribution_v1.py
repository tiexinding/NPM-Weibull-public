"""Paper#5 supplement S12: full distribution of the identity-axis scale fields (beyond the width H^W).

For every matrix the centred fields are
    h^r_i = ln RMS_i(W) - mean_i,   h^c_j = ln RMS_j(W) - mean_j      (raw W, no normalization)
and per field we record: sd (= H^W_row / H^W_col, must reproduce p5v2_twoaxis_bridge_v1.json bitwise),
IQR, skewness, excess kurtosis, 5%/95% quantiles, participation ratio
    PR = (sum e^{2h})^2 / (n sum e^{4h})            (effective fraction of active channels),
and two bimodality read-outs: Hartigan dip statistic + p-value (diptest, if importable) and Sarle's
bimodality coefficient BC = (skew^2 + 1) / (kurt_excess + 3 (n-1)^2 / ((n-2)(n-3))), BC > 5/9 flags
possible bimodality.  Row fields for q/k/v/gate/up, column fields for v/o/down (v both).

Sources (read-only):
  04_tclock_runs/wt_ckpts/tclock_D{1..4}_s{1,2,3}/step{800,1600,3200,5000}.pt   (48 ckpt x 42 matrices)
  04_ea_runs/ckpt_key/ea_{gaussian,laplace,uniform}_s{1,2,3}/ckpt/step{0,...,30000}.pt (63 x 42)
  05_ivn_runs/ivn_{base,ropeperm}/ckpt/step30000.pt                                   (2 x 42)
  08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json   (H_row/H_col cross-check, TCLOCK)
Outputs:
  08_paper5_draft/data/p5v2_field_distribution_v1.json
  08_paper5_draft/figures/P5v2_S12_field_distribution.png
"""
import hashlib
import json
import re
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skew, kurtosis

try:
    import diptest
    HAVE_DIP = True
except Exception:  # pragma: no cover
    HAVE_DIP = False

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "04_tclock_runs/wt_ckpts"
CK = ROOT / "04_ea_runs/ckpt_key"
IVN = ROOT / "05_ivn_runs"
TWOAXIS = ROOT / "08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json"
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_field_distribution_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S12_field_distribution.png"

ARMS, SEEDS, STEPS = ["D1", "D2", "D3", "D4"], [1, 2, 3], [800, 1600, 3200, 5000]
INITS, EA_STEPS = ["gaussian", "laplace", "uniform"], [0, 800, 3200, 10000, 20000, 25000, 30000]
FAM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
ROW_FIELD = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
COL_FIELD = {"v_proj", "o_proj", "down_proj"}
IDENTITY_SIDE = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "o_proj": "col", "gate_proj": "row", "up_proj": "row", "down_proj": "col"}
BC_THRESHOLD = 5.0 / 9.0     # Sarle's bimodality coefficient reference (uniform distribution)


def field_stats(h):
    h = np.asarray(h, dtype=float); n = h.size
    q05, q25, q75, q95 = np.quantile(h, [0.05, 0.25, 0.75, 0.95])
    sk = float(skew(h)); ku = float(kurtosis(h))                      # excess kurtosis
    pr = float((np.exp(2 * h).sum() ** 2) / (n * np.exp(4 * h).sum()))
    bc = float((sk ** 2 + 1) / (ku + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    d = {"n": int(n), "sd": float(h.std()), "iqr": float(q75 - q25), "skew": sk, "kurt_excess": ku,
         "q05": float(q05), "q95": float(q95), "min": float(h.min()), "max": float(h.max()), "PR": pr, "BC": bc}
    if HAVE_DIP:
        dip, p = diptest.diptest(h)
        d["dip"] = float(dip); d["dip_p"] = float(p)
    return d


def analyse(w):
    hr = np.log(np.sqrt((w ** 2).mean(1))); hc = np.log(np.sqrt((w ** 2).mean(0)))
    return {"row": field_stats(hr - hr.mean()), "col": field_stats(hc - hc.mean()),
            "row_field": (hr - hr.mean()).tolist(), "col_field": (hc - hc.mean()).tolist()}


def do_state(st, meta, keep_fields):
    rows = []
    for nm, w in st.items():
        mm = re.search(r"\.(\w+_proj)\.weight$", nm)
        if not mm or mm.group(1) not in FAM:
            continue
        L = int(re.search(r"layers\.(\d+)", nm).group(1))
        d = analyse(w.double().numpy())
        r = {"row": d["row"], "col": d["col"], "kind": mm.group(1), "layer": L, "mat": nm, "shape": list(w.shape)}
        if keep_fields(meta, mm.group(1), L):
            r["row_field"] = d["row_field"]; r["col_field"] = d["col_field"]
        r.update(meta); rows.append(r)
    assert len(rows) == 42, (meta, len(rows))
    return rows


# representative fields kept for the histogram panel
def keep_tclock(meta, kind, L):
    return meta["seed"] == 1 and meta["step"] == 5000 and (
        (kind == "q_proj" and L == 3 and meta["arm"] in ("D1", "D3")) or (kind == "gate_proj" and L == 0 and meta["arm"] == "D3")
        or (kind == "v_proj" and L == 3 and meta["arm"] == "D1"))


def do_tclock(args):
    arm, sd, step = args
    st = torch.load(WT / f"tclock_{arm}_s{sd}/step{step}.pt", map_location="cpu", weights_only=True)
    return do_state(st, {"set": "tclock", "arm": arm, "seed": sd, "step": step}, keep_tclock)


def do_ea(args):
    init, sd, step = args
    st = torch.load(CK / f"ea_{init}_s{sd}/ckpt/step{step}.pt", map_location="cpu", weights_only=True)
    return do_state(st, {"set": "ea", "init": init, "seed": sd, "step": step}, lambda m, k, L: False)


def do_ivn(arm):
    st = torch.load(IVN / f"ivn_{arm}/ckpt/step30000.pt", map_location="cpu", weights_only=True)
    return do_state(st, {"set": "ivn", "arm": arm, "seed": 1, "step": 30000}, lambda m, k, L: False)


def side_stats(r):
    """identity-axis field stats of a row (row for q/k/gate/up, col for o/down, row for v; v col separately)."""
    return r["row"] if IDENTITY_SIDE[r["kind"]] == "row" else r["col"]


def med(vals):
    return float(np.median(vals)) if len(vals) else float("nan")


STATS = ["sd", "iqr", "skew", "kurt_excess", "q05", "q95", "PR", "BC"] + (["dip", "dip_p"] if HAVE_DIP else [])


def summarize(rows, keys):
    """median of each stat over rows grouped by the given meta keys (kind always included), identity side + v col."""
    out = {}
    groups = sorted(set(tuple(r[k] for k in keys) for r in rows))
    for g in groups:
        sub = [r for r in rows if tuple(r[k] for k in keys) == g]
        for f in FAM:
            ss = [r for r in sub if r["kind"] == f]
            if not ss:
                continue
            e = {s: med([side_stats(r)[s] for r in ss]) for s in STATS}
            e["n"] = len(ss); e["side"] = IDENTITY_SIDE[f]
            if HAVE_DIP:
                e["frac_dip_p_below_0.05"] = float(np.mean([side_stats(r)["dip_p"] < 0.05 for r in ss]))
            e["frac_BC_above_5_9"] = float(np.mean([side_stats(r)["BC"] > BC_THRESHOLD for r in ss]))
            out["|".join(map(str, g)) + "|" + LAB[f]] = e
            if f == "v_proj":
                e2 = {s: med([r["col"][s] for r in ss]) for s in STATS}; e2["n"] = len(ss); e2["side"] = "col"
                out["|".join(map(str, g)) + "|v_col"] = e2
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        tc = [r for rr in pool.map(do_tclock, list(product(ARMS, SEEDS, STEPS))) for r in rr]
        ea = [r for rr in pool.map(do_ea, list(product(INITS, SEEDS, EA_STEPS))) for r in rr]
    iv = [r for rr in map(do_ivn, ["base", "ropeperm"]) for r in rr]
    assert len(tc) == 48 * 42 and len(ea) == 63 * 42 and len(iv) == 2 * 42, (len(tc), len(ea), len(iv))

    # ---- cross-check sd against the two-axis bridge record (TCLOCK, bitwise)
    ta = json.load(open(TWOAXIS))["per_matrix"]
    key = {(r["arm"], r["seed"], r["step"], r["mat"]): r for r in ta}
    dev = []
    for r in tc:
        a = key[(r["arm"], r["seed"], r["step"], r["mat"])]
        dev.append(max(abs(r["row"]["sd"] - a["H_row"]), abs(r["col"]["sd"] - a["H_col"])))
    assert len(dev) == 48 * 42 and max(dev) < 1e-9, max(dev)

    # ---- summaries
    tc_kas = summarize(tc, ["arm", "step"])                       # kind x arm x step
    tc_ks = summarize(tc, ["step"])                               # kind x step (all arms)
    tc_k5000 = summarize([r for r in tc if r["step"] == 5000], ["set"])
    ea_kis = summarize(ea, ["init", "step"])                      # kind x init x step
    iv_k = summarize(iv, ["arm"])

    # ---- same-H, different-shape examples: pairs of matrices (same kind) with |ΔH| < 0.005 and largest |Δskew| or BC gap
    ex = []
    for f in FAM:
        ss = [r for r in tc if r["kind"] == f]
        best = None
        for i in range(len(ss)):
            si = side_stats(ss[i])
            for j in range(i + 1, len(ss)):
                sj = side_stats(ss[j])
                if abs(si["sd"] - sj["sd"]) < 0.005 and si["sd"] > 0.10:
                    gap = abs(si["skew"] - sj["skew"])
                    if best is None or gap > best[0]:
                        best = (gap, ss[i], ss[j])
        if best:
            (gap, a, b) = best
            ex.append({"kind": LAB[f], "H": [side_stats(a)["sd"], side_stats(b)["sd"]], "skew": [side_stats(a)["skew"], side_stats(b)["skew"]],
                       "kurt_excess": [side_stats(a)["kurt_excess"], side_stats(b)["kurt_excess"]], "BC": [side_stats(a)["BC"], side_stats(b)["BC"]],
                       "PR": [side_stats(a)["PR"], side_stats(b)["PR"]],
                       "a": f"{a['arm']} s{a['seed']} step{a['step']} L{a['layer']}", "b": f"{b['arm']} s{b['seed']} step{b['step']} L{b['layer']}"})

    hist_examples = {f"{r['arm']}|{LAB[r['kind']]}|L{r['layer']}": {"row_field": r["row_field"], "col_field": r["col_field"], "row": r["row"], "col": r["col"]}
                     for r in tc if "row_field" in r}
    assert len(hist_examples) == 4, list(hist_examples)

    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("row_field", "col_field")} for r in rows]
    out = {"schema": "p5v2-field-distribution-v1", "script": Path(__file__).name,
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
           "protocol": "centred fields h = ln RMS - mean(ln RMS) on raw W; row field for q/k/v/gate/up, column field for v/o/down; "
                       "stats: sd (=H^W), IQR, skew, excess kurtosis, q05/q95, PR=(sum e^2h)^2/(n sum e^4h), Sarle BC (>5/9 flags), "
                       + ("Hartigan dip + p (diptest)" if HAVE_DIP else "dip test unavailable"),
           "bimodality_method": "diptest (Hartigan dip, p-value) and Sarle BC" if HAVE_DIP else "Sarle BC only (diptest not importable)",
           "BC_threshold": BC_THRESHOLD, "H_crosscheck_max_abs_dev": max(dev), "H_crosscheck_n": len(dev),
           "n_matrices": {"tclock": len(tc), "ea": len(ea), "ivn": len(iv)},
           "tclock_kind_arm_step": tc_kas, "tclock_kind_step": tc_ks, "tclock_kind_step5000_allarms": tc_k5000,
           "ea_kind_init_step": ea_kis, "ivn_kind_arm": iv_k,
           "same_H_different_shape_examples": ex, "hist_examples": hist_examples,
           "per_matrix": {"tclock": strip(tc), "ea": strip(ea), "ivn": strip(iv)}}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_JSON)

    # =============================== figure ===============================
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
    C_ARM = {"D1": "#2166AC", "D2": "#67A9CF", "D3": "#B2182B", "D4": "#EF8A62"}
    C_INIT = {"gaussian": "0.25", "laplace": "#B2182B", "uniform": "#2166AC"}
    fig = plt.figure(figsize=(14.4, 8.6))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.36)
    xs = np.arange(len(FAM))

    # ---- (a) skew and kurtosis per kind, TCLOCK 5000, arms coloured
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()
    for i, f in enumerate(FAM):
        for arm in ARMS:
            ss = [side_stats(r) for r in tc if r["kind"] == f and r["step"] == 5000 and r["arm"] == arm]
            sk = [s["skew"] for s in ss]; ku = [s["kurt_excess"] for s in ss]
            off = (ARMS.index(arm) - 1.5) * 0.12
            ax.scatter(np.full(len(sk), i - 0.18 + off), sk, s=9, color=C_ARM[arm], alpha=0.5, lw=0)
            ax.plot([i - 0.18 + off - 0.04, i - 0.18 + off + 0.04], [np.median(sk)] * 2, color=C_ARM[arm], lw=2.2)
            ax2.scatter(np.full(len(ku), i + 0.22 + off), ku, s=9, color=C_ARM[arm], alpha=0.5, lw=0, marker="s")
            ax2.plot([i + 0.22 + off - 0.04, i + 0.22 + off + 0.04], [np.median(ku)] * 2, color=C_ARM[arm], lw=2.2, ls=(0, (2, 1)))
    ax.axhline(0, color="0.6", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([LAB[f] for f in FAM])
    ax.set_ylabel("skewness of identity-axis field  (circles, left)"); ax2.set_ylabel("excess kurtosis  (squares, right)")
    ax.set_title("(a) Skewness and kurtosis of the identity-axis field, grid at step 5,000", loc="left")
    ax.legend(handles=[Line2D([], [], color=C_ARM[a], lw=2.2, label=a) for a in ARMS] + [Line2D([], [], color="0.4", lw=2.2, label="median (18 matrices)")],
              loc="upper left", frameon=False, ncol=5, columnspacing=0.8, handlelength=1.4)

    # ---- (b) PR over training, E-A three initializations, per kind
    ax = fig.add_subplot(gs[0, 1])
    def _sx(v):
        v = np.asarray(v, float); return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))
    MK = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "o_proj": "P", "gate_proj": "D", "up_proj": "v", "down_proj": "X"}
    for f in FAM:
        for init in INITS:
            y = [ea_kis[f"{init}|{s}|{LAB[f]}"]["PR"] for s in EA_STEPS]
            ax.plot(_sx(EA_STEPS), y, color=C_INIT[init], marker=MK[f], ms=3.6, lw=1.0, alpha=0.9, mec="white", mew=0.4)
    ax.set_xticks(_sx([0, 800, 3200, 10000, 30000])); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
    ax.set_xlabel("training step  (linear to 800, log above)"); ax.set_ylabel("participation ratio of identity-axis field")
    ax.set_title("(b) Effective fraction of active channels over training, E-A runs", loc="left")
    hb = [Line2D([], [], color=C_INIT[i], lw=1.6, label=f"{i} init") for i in INITS] + \
         [Line2D([], [], color="0.4", marker=MK[f], ls="", ms=4, label=LAB[f]) for f in FAM]
    ax.legend(handles=hb, loc="lower left", frameon=False, ncol=4, columnspacing=0.8, handlelength=1.4)

    # ---- (c) representative histograms
    ax = fig.add_subplot(gs[1, 0])
    order = [("D1|q|L3", "row_field", "q L3 row, D1 (structured)", "#2166AC", "-"), ("D3|q|L3", "row_field", "q L3 row, D3 (shuffled high-rep.)", "#B2182B", "-"),
             ("D3|gate|L0", "row_field", "gate L0 row, D3", "#EF8A62", "-"), ("D1|v|L3", "row_field", "v L3 row, D1", "0.3", "-"), ("D1|v|L3", "col_field", "v L3 column, D1", "0.3", "--")]
    for k, fld, lab, c, ls in order:
        h = np.array(hist_examples[k][fld]); st = hist_examples[k]["row" if fld == "row_field" else "col"]
        bins = np.linspace(-1.0, 0.8, 61)
        ax.hist(h, bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.4,
                label=f"{lab}: H {st['sd']:.2f}, skew {st['skew']:+.2f}, PR {st['PR']:.2f}, BC {st['BC']:.2f}")
    ax.set_xlabel("centred field  $h = \\ln \\mathrm{RMS} - \\overline{\\ln \\mathrm{RMS}}$"); ax.set_ylabel("density")
    ax.set_title("(c) Representative fields, seed 1, step 5,000", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.4)

    # ---- (d) bimodality read-out per kind x arm
    ax = fig.add_subplot(gs[1, 1])
    for i, f in enumerate(FAM):
        for arm in ARMS:
            e = tc_kas[f"{arm}|5000|{LAB[f]}"]
            off = (ARMS.index(arm) - 1.5) * 0.18
            ax.bar(i + off, e["BC"], width=0.16, color=C_ARM[arm], alpha=0.85)
            if HAVE_DIP:
                ax.text(i + off, e["BC"] + 0.01, f"{e['frac_dip_p_below_0.05']:.0%}", ha="center", va="bottom", fontsize=6.2, color="0.25", rotation=90)
    ax.axhline(BC_THRESHOLD, color="0.35", lw=1.0, ls=(0, (5, 3)))
    ax.text(len(FAM) - 0.5, BC_THRESHOLD + 0.01, "BC = 5/9", ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_xticks(xs); ax.set_xticklabels([LAB[f] for f in FAM]); ax.set_ylabel("Sarle bimodality coefficient (median over 18 matrices)")
    ax.set_title("(d) Bimodality read-out per kind and arm, step 5,000" + (" (text: share of matrices with dip p < 0.05)" if HAVE_DIP else ""), loc="left", fontsize=9.6)
    ax.legend(handles=[Line2D([], [], color=C_ARM[a], lw=6, label=a) for a in ARMS], loc="upper left", frameon=False, ncol=4)
    for a in fig.axes:
        a.spines[["top"]].set_visible(False)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT_FIG)

    # ---- console summary
    print("dip available:", HAVE_DIP)
    for f in FAM:
        e = tc_k5000[f"tclock|{LAB[f]}"]
        print(f"{LAB[f]:5s} side={e['side']} H {e['sd']:.3f} skew {e['skew']:+.2f} kurt {e['kurt_excess']:+.2f} PR {e['PR']:.3f} BC {e['BC']:.2f}"
              + (f" dip_p<.05 {e['frac_dip_p_below_0.05']:.0%}" if HAVE_DIP else ""))
    for f in FAM:
        print(LAB[f], {arm: (round(tc_kas[f'{arm}|5000|{LAB[f]}']['skew'], 2), round(tc_kas[f'{arm}|5000|{LAB[f]}']['PR'], 3), round(tc_kas[f'{arm}|5000|{LAB[f]}']['BC'], 2)) for arm in ARMS})
    for e in ex:
        print("sameH", e)
