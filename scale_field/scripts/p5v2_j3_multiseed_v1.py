#!/usr/bin/env python3
"""Per-seed J3 verdict for the RoPE-map permutation (Paper#5 K4/K5), seeds 1,2,(3).

Reuses cum_rel_drift / pair_avg / freq_profile from scripts/ivn_j1_j3_hw_v1.py (paths generalized)
and the Weibull protocol (kfit / rownorm / H^W sd) from scripts/p5v2_twoaxis_bridge_v1.py.

Seed 1: base = 05_ivn_runs/ivn_base, perm = 05_ivn_runs/ivn_ropeperm   (pre-registered record 05_ivn_runs/ivn_j1_j3_hw_v1.json)
Seed s: base = 04_ea_runs/ckpt_key/ea_gaussian_s{s}  (same config as IVN base; batch stream by seed)
        perm = 10_p5batch_cloud/results/ropeperm_s{s}  (cloud batch 260911)

Two profile conventions are reported:
  full    : STEPS_J1 = [3200,5000,10000,15000,20000,25000,30000]  (only seed 1 has this grid)
  reduced : COMMON  = [3200,10000,30000]  (the checkpoints shared by the E-A base and the cloud ropeperm runs)
Seed 1 is computed under both; the pre-registered numbers are asserted to reproduce under 'full'.

Verdict per seed (layers are NOT pooled as replicates):
  q, k : dC = C_freq - C_coord > 0 in 6/6 layers AND pooled dC > permutation-null p95
  v    : negative control, pooled dC inside the null
K5 globals per seed: terminal k_raw / k_norm(row) per kind (relative change perm vs base, %),
  H^W = sd_i(ln RMS_i) change (%), median over the 6 layers of the terminal checkpoint.
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ivn_j1_j3_hw_v1 import cum_rel_drift, freq_profile, STEPS_J1, NF, ROW_KINDS  # noqa: E402
from p5v2_twoaxis_bridge_v1 import kfit, rownorm  # noqa: E402

COMMON = [3200, 10000, 30000]
KINDS7 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
N_NULL = 2000


def run_dirs(seed):
    if seed == 1:
        return (ROOT / "05_ivn_runs/ivn_base/ckpt", ROOT / "05_ivn_runs/ivn_ropeperm/ckpt",
                ROOT / "05_ivn_runs/ivn_ropeperm/run_meta.json")
    return (ROOT / f"04_ea_runs/ckpt_key/ea_gaussian_s{seed}/ckpt",
            ROOT / f"10_p5batch_cloud/results/ropeperm_s{seed}/ckpt",
            ROOT / f"10_p5batch_cloud/results/ropeperm_s{seed}/run_meta.json")


def load_pi(meta_path):
    m = json.load(open(meta_path))
    pi = m.get("pi")
    if pi is None:
        pf = json.load(open(meta_path.parent / "preflight.json"))
        pi = (pf.get("preflight", pf)).get("pi")
    pi = np.asarray(pi, dtype=int)
    assert sorted(pi.tolist()) == list(range(NF)), "pi is not a permutation of 0..31"
    inv = np.empty(NF, dtype=int); inv[pi] = np.arange(NF)
    return pi, inv, m


def j3_one(Xb, Xp, pi_inv, rng_seed=0):
    out = {}
    for kd in ROW_KINDS:
        Pb, Pc = freq_profile(Xb, kd), freq_profile(Xp, kd)
        Pf = Pc[pi_inv]
        c_f = float(np.corrcoef(Pf, Pb)[0, 1]); c_c = float(np.corrcoef(Pc, Pb)[0, 1])
        Lb, Lc = freq_profile(Xb, kd, True), freq_profile(Xp, kd, True)
        per_layer = {}
        for L in sorted(Lb):
            pf = float(np.corrcoef(Lc[L][pi_inv], Lb[L])[0, 1]); pc = float(np.corrcoef(Lc[L], Lb[L])[0, 1])
            per_layer[int(L)] = {"C_freq": pf, "C_coord": pc, "dC": pf - pc}
        rng = np.random.default_rng(rng_seed)
        null = np.array([float(np.corrcoef(Pc[rng.permutation(NF)], Pb)[0, 1]) - c_c for _ in range(N_NULL)])
        # per-layer null band (same construction, per layer) for the figure
        null_layer = {}
        for L in sorted(Lb):
            rl = np.random.default_rng(rng_seed + 1 + int(L))
            nl = np.array([float(np.corrcoef(Lc[L][rl.permutation(NF)], Lb[L])[0, 1]) - per_layer[int(L)]["C_coord"] for _ in range(500)])
            null_layer[int(L)] = float(np.percentile(nl, 95))
        dC = c_f - c_c
        n_pos = sum(1 for v in per_layer.values() if v["dC"] > 0)
        n_over = sum(1 for L, v in per_layer.items() if v["dC"] > null_layer[L])
        out[kd] = {"C_freq": c_f, "C_coord": c_c, "dC": dC, "per_layer": per_layer,
                   "n_layers_dC_pos": n_pos, "n_layers_dC_over_layer_null_p95": n_over,
                   "null_dC_p95": float(np.percentile(null, 95)), "null_dC_mean": float(np.mean(null)),
                   "null_layer_p95": null_layer,
                   "dC_exceeds_null_p95": bool(dC > np.percentile(null, 95)),
                   "null_dC_p05": float(np.percentile(null, 5)),
                   "dC_below_null_p05": bool(dC < np.percentile(null, 5)),   # two-sided reading for the v control: field stays at coordinate
                   "is_negative_control": kd == "v_proj"}
    return out


def verdict(j3):
    q, k, v = j3["q_proj"], j3["k_proj"], j3["v_proj"]
    ok_q = q["n_layers_dC_pos"] == 6 and q["dC_exceeds_null_p95"]
    ok_k = k["n_layers_dC_pos"] == 6 and k["dC_exceeds_null_p95"]
    ok_v = not v["dC_exceeds_null_p95"]
    return {"q_pass": ok_q, "k_pass": ok_k, "v_negative_control_pass": ok_v, "all_pass": ok_q and ok_k and ok_v}


def terminal_globals(base_dir, perm_dir, step=30000):
    """K5: per kind, median over 6 layers of k_raw, k_row (row-normalized refit), H^W=sd(ln rowRMS)."""
    def per_kind(ckpt_dir):
        sd = torch.load(os.path.join(ckpt_dir, f"step{step}.pt"), map_location="cpu", weights_only=True)
        acc = {kd: {"k_raw": [], "k_row": [], "H_row": [], "r2": []} for kd in KINDS7}
        for nm, tw in sd.items():
            kd = nm.rsplit(".", 2)[-2]
            if kd not in acc:
                continue
            w = tw.to(torch.float64).numpy()
            kr, r2r = kfit(w); kw, r2w = kfit(rownorm(w))
            acc[kd]["k_raw"].append(kr); acc[kd]["k_row"].append(kw); acc[kd]["r2"].append(min(r2r, r2w))
            acc[kd]["H_row"].append(float(np.log(np.sqrt((w ** 2).mean(1))).std()))
        return {kd: {m: float(np.median(v)) for m, v in d.items()} for kd, d in acc.items()}
    B, P = per_kind(base_dir), per_kind(perm_dir)
    out = {}
    for kd in KINDS7:
        out[kd] = {"base": B[kd], "perm": P[kd],
                   "k_raw_pct": 100 * (P[kd]["k_raw"] / B[kd]["k_raw"] - 1),
                   "k_row_pct": 100 * (P[kd]["k_row"] / B[kd]["k_row"] - 1),
                   "H_row_pct": 100 * (P[kd]["H_row"] / B[kd]["H_row"] - 1),
                   "min_r2": min(B[kd]["r2"], P[kd]["r2"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out_json", default=str(ROOT / "10_p5batch_cloud/results/j3_multiseed_v1.json"))
    ap.add_argument("--out_png", default=str(ROOT / "10_p5batch_cloud/results/j3_multiseed_v1.png"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    pre = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
    res = {"schema": "p5v2-j3-multiseed-v1", "script_sha256_16": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
           "conventions": {"full": STEPS_J1, "reduced": COMMON,
                           "note": "profile = cumulative median-centred delta ln(row RMS) over consecutive checkpoints of the convention, "
                                   "rotate_half pair-averaged (j, j+32), median over 8 heads x 6 layers (pooled) or over heads (per layer)"},
           "verdict_rule": "q,k: dC>0 in 6/6 layers and pooled dC > permutation-null p95 (2000 draws); v: pooled dC inside null",
           "seeds": {}}

    for s in seeds:
        bdir, pdir, meta = run_dirs(s)
        for d in (bdir, pdir):
            for st in COMMON:
                assert (Path(d) / f"step{st}.pt").exists(), f"missing {d}/step{st}.pt"
        pi, pi_inv, m = load_pi(meta)
        entry = {"base_dir": str(bdir), "perm_dir": str(pdir), "pi": pi.tolist(), "pi_sha256_16": hashlib.sha256(pi.tobytes()).hexdigest()[:16]}
        if s == 1:
            assert pi.tolist() == pre["pi"], "seed-1 pi differs from pre-registered record"
            Xb, Xp = cum_rel_drift(str(bdir), STEPS_J1), cum_rel_drift(str(pdir), STEPS_J1)
            j3_full = j3_one(Xb, Xp, pi_inv)
            for kd in ROW_KINDS:
                for key in ("C_freq", "C_coord", "dC"):
                    dev = abs(j3_full[kd][key] - pre["J3"][kd][key])
                    assert dev < 1e-6, (kd, key, dev)
                for L in range(6):
                    for key in ("C_freq", "C_coord"):
                        dev = abs(j3_full[kd]["per_layer"][L][key] - pre["J3"][kd]["per_layer"][str(L)][key])
                        assert dev < 1e-6, (kd, L, key, dev)
            entry["full"] = {"J3": j3_full, "verdict": verdict(j3_full), "reproduces_prereg_1e-6": True}
        else:
            assert pi.tolist() == pre["pi"], f"seed {s} used a different pi than seed 1"
        Xb, Xp = cum_rel_drift(str(bdir), COMMON), cum_rel_drift(str(pdir), COMMON)
        j3_red = j3_one(Xb, Xp, pi_inv)
        entry["reduced"] = {"J3": j3_red, "verdict": verdict(j3_red)}
        entry["K5_terminal"] = terminal_globals(str(bdir), str(pdir))
        # batch-stream pairing check (first-5 hashes) when both metas exist
        try:
            bm = json.load(open(Path(bdir).parent / "run_meta.json"))
            entry["batch_hash_first5_equal"] = bm.get("batch_hashes_first5") == m.get("batch_hashes_first5")
        except Exception:
            entry["batch_hash_first5_equal"] = None
        res["seeds"][str(s)] = entry

    if "1" in res["seeds"] and "full" in res["seeds"]["1"]:
        f, r = res["seeds"]["1"]["full"]["J3"], res["seeds"]["1"]["reduced"]["J3"]
        res["seed1_convention_effect"] = {kd: {k: {"full": f[kd][k], "reduced": r[kd][k]} for k in ("C_freq", "C_coord", "dC")} for kd in ROW_KINDS}

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=1)

    # ---------------- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "figure.dpi": 110})
    C = {"q_proj": "#B2182B", "k_proj": "#EF8A62", "v_proj": "#808080"}
    ns = len(seeds)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.3})
    ax = axes[0]
    xoff = {kd: i for i, kd in enumerate(ROW_KINDS)}
    for si, s in enumerate(seeds):
        j3 = res["seeds"][str(s)]["reduced"]["J3"]
        for kd in ROW_KINDS:
            xs = np.array([si * 8 + xoff[kd] * 2.2 + L * 0.3 for L in range(6)])
            ys = [j3[kd]["per_layer"][L]["dC"] for L in range(6)]
            nl = [j3[kd]["null_layer_p95"][L] for L in range(6)]
            ax.plot(xs, ys, "o", color=C[kd], ms=4.2, mec="white", mew=0.5, zorder=4, label=f"{LAB[kd]}" if si == 0 else None)
            ax.plot(xs, nl, "_", color="0.55", ms=8, mew=1.2, zorder=3, label="per-layer permutation null 95%" if (si == 0 and kd == "q_proj") else None)
        ax.axvline(si * 8 - 1.2, color="0.85", lw=0.8)
        ax.text(si * 8 + 2.6, 1.02, f"seed {s}", ha="center", va="bottom", fontsize=9, color="0.3")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylim(-0.6, 1.1); ax.set_xticks([]); ax.set_ylabel(r"identity advantage  $\Delta C = C_{\mathrm{freq}} - C_{\mathrm{coord}}$")
    ax.set_title("(a) Per-layer identity advantage per seed (reduced convention 3.2k/10k/30k)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax = axes[1]
    w = 0.38
    for si, s in enumerate(seeds):
        j3 = res["seeds"][str(s)]["reduced"]["J3"]
        for ki, kd in enumerate(ROW_KINDS):
            x = si * 3.6 + ki
            ax.bar(x - w / 2, j3[kd]["C_freq"], w, color=C[kd], label=("re-indexed by frequency" if (si == 0 and ki == 0) else None))
            ax.bar(x + w / 2, j3[kd]["C_coord"], w, color="0.75", label=("kept at coordinate" if (si == 0 and ki == 0) else None))
            ax.text(x, max(j3[kd]["C_freq"], j3[kd]["C_coord"]) + 0.02, LAB[kd], ha="center", fontsize=8)
        ax.text(si * 3.6 + 1, 1.06, f"seed {s}", ha="center", fontsize=9, color="0.3")
    ax.set_ylim(-0.3, 1.15); ax.set_xticks([]); ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("profile similarity, permuted vs base"); ax.set_title("(b) Pooled similarity per seed", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
    fig.savefig(a.out_png, dpi=200, bbox_inches="tight", facecolor="white")

    # ---------------- console
    for s in seeds:
        e = res["seeds"][str(s)]
        for conv in ("full", "reduced"):
            if conv not in e:
                continue
            j3, vd = e[conv]["J3"], e[conv]["verdict"]
            print(f"seed {s} [{conv}] " + " | ".join(
                f"{LAB[kd]}: Cf {j3[kd]['C_freq']:+.3f} Cc {j3[kd]['C_coord']:+.3f} dC {j3[kd]['dC']:+.3f} "
                f"pos {j3[kd]['n_layers_dC_pos']}/6 >null {j3[kd]['n_layers_dC_over_layer_null_p95']}/6 null95 {j3[kd]['null_dC_p95']:.3f}"
                for kd in ROW_KINDS) + f"  => {vd}")
        g = e["K5_terminal"]
        print(f"seed {s} K5 terminal (perm vs base, %): " + " ".join(
            f"{LAB[kd]} k_raw {g[kd]['k_raw_pct']:+.2f} k_row {g[kd]['k_row_pct']:+.2f} H {g[kd]['H_row_pct']:+.1f} (r2min {g[kd]['min_r2']:.3f})" for kd in KINDS7))
        print(f"seed {s} batch_hash_first5_equal(base,perm): {e.get('batch_hash_first5_equal')}")
    if "seed1_convention_effect" in res:
        print("seed 1 convention effect:", json.dumps(res["seed1_convention_effect"]))
    print("wrote", a.out_json, a.out_png)


if __name__ == "__main__":
    main()
