#!/usr/bin/env python3
"""IVN 补做 spec v2 §1「强制记录的次级读数」中 v2 报告仍欠的项 (离线, 零 GPU)。

覆盖 4 项 (第 5 项 held-out 另见 ivn_heldout_loss_v1.py):
  A. update exposure ratio  ‖u^干预‖/‖u^Adam‖ 行级         [flatv 训练时已记录; nomom 见 §限制]
  B. alignment / injection  行级 η⟨u,w⟩ 累计 + 无量纲对齐   [mv/energy npz]
  C. row-normalization rescue k (行归一后 k) + 列归一轴特异性对照  [ckpt 重算, 复用主拟合核]
  D. progress-matched descriptive sensitivity               [等 loss 点重算 J2; 非因果判读]

口径与既有资产对齐:
  - C 复用 scripts/row_rescue_v1.py 的 row_normalize / fit_kd (其拟合核 = opt_offline_analysis
    .weibull_fit_1024 逐字同构), 但**按 q/k 拆开**聚合 (spec P0-4: pooled qk 只作摘要量)。
  - B 的 Au 即训练时累计的 au_inc = Σ_t η_t⟨u_t, w_t⟩_row (ivn_train.py: au_inc =
    -(d_adp*W_pre).sum(1), d_adp = -η_t u_t) — 严格量, 无重建误差。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from row_rescue_v1 import row_normalize, fit_kd          # 复用主拟合核


def col_normalize(w):
    """列归一 = 转置后行归一。用于轴特异性对照: q/k/v 的 head 结构在行, o 的在列
    (t123_structure_v1.py 口径), 故 o 应被列归一而非行归一救回。"""
    return row_normalize(w.T).T

ARMS = ["base", "flatv", "nomom", "ropeperm"]
TERM = [20000, 25000, 30000]
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FFN = ["gate_proj", "up_proj", "down_proj"]


def _kind(nm):
    return nm.rsplit(".", 2)[-2]


# ---------------- A. exposure ratio ----------------
def exposure(run):
    """‖u^干预‖/‖u^Adam‖ 行级 (训练时从同一 m/v state 记录, 无重建误差)。"""
    out = {}
    for s in TERM:
        d = np.load(f"{run}/ckpt/mv_step{s}.npz")
        ks = [k for k in d.files if k.endswith("|exposure_ratio")]
        if not ks:
            return None
        per = {}
        for k in ks:
            per.setdefault(_kind(k.split("|")[0]), []).append(d[k].astype(np.float64))
        out[s] = {kd: {"median": float(np.median(np.concatenate(v))),
                       "p05": float(np.percentile(np.concatenate(v), 5)),
                       "p95": float(np.percentile(np.concatenate(v), 95))}
                  for kd, v in per.items()}
    return out


# ---------------- B. alignment / injection ----------------
def align_inject(run):
    """Au = Σ_t η_t⟨u_t,w_t⟩_row (累计注入, 严格);
    无量纲对齐 α = Au / sqrt(Σ_t η_t²‖u_t‖²_row) / ‖w_row‖  — **累计口径**, 非逐步 ⟨û,ŵ⟩ 均值。"""
    out = {}
    for s in TERM:
        mv = np.load(f"{run}/ckpt/mv_step{s}.npz")
        en = np.load(f"{run}/ckpt/energy_step{s}.npz")
        sd = torch.load(f"{run}/ckpt/step{s}.pt", map_location="cpu", weights_only=True)
        per_inj, per_al = {}, {}
        for nm, tw in sd.items():
            kd = _kind(nm)
            au = mv[f"{nm}|Au"].astype(np.float64)
            adp = en[f"{nm}|adp|row"].astype(np.float64)          # Σ η²‖u‖²_row
            wrow = np.sqrt((tw.numpy().astype(np.float64) ** 2).sum(1))
            al = au / np.maximum(np.sqrt(adp) * wrow, 1e-30)
            per_inj.setdefault(kd, []).append(au)
            per_al.setdefault(kd, []).append(al)
        out[s] = {kd: {"injection_median": float(np.median(np.concatenate(per_inj[kd]))),
                       "alignment_cum_median": float(np.median(np.concatenate(per_al[kd])))}
                  for kd in per_inj}
    return out


# ---------------- C. row-normalization rescue k ----------------
def rescue(run):
    """终端窗 3 ckpt: 逐矩阵 before/after 行归一拟合 → per-kind 中位 → 三步中位 (q/k 拆开)。"""
    per_step = []
    for s in TERM:
        sd = torch.load(f"{run}/ckpt/step{s}.pt", map_location="cpu", weights_only=True)
        cur = {}
        for nm, tw in sd.items():
            w = tw.numpy().astype(np.float64)
            wn = row_normalize(w)
            # 断言: 行归一保持全矩阵 RMS (各行等长 → 解析成立)
            r0, r1 = np.sqrt((w ** 2).mean()), np.sqrt((wn ** 2).mean())
            assert abs(r1 - r0) / r0 < 1e-12, f"row_normalize 未保 RMS: {nm} {r0} {r1}"
            cur.setdefault(_kind(nm), []).append(
                (fit_kd(w)[0], fit_kd(wn)[0], fit_kd(col_normalize(w))[0]))
        per_step.append({kd: (float(np.median([x[0] for x in v])),
                              float(np.median([x[1] for x in v])),
                              float(np.median([x[2] for x in v]))) for kd, v in cur.items()})
    return {kd: {"k_before": float(np.median([ps[kd][0] for ps in per_step])),
                 "k_after_row": float(np.median([ps[kd][1] for ps in per_step])),
                 "k_after_col": float(np.median([ps[kd][2] for ps in per_step]))}
            for kd in per_step[0]}


# ---------------- D. progress-matched sensitivity ----------------
def loss_at(run, step, half_win=250):
    L = [json.loads(l) for l in open(f"{run}/loss.jsonl")]
    w = [x["loss"] for x in L if abs(x.get("step", -1) - step) <= half_win]
    return float(np.median(w)) if w else None


def k_at_step(run_json, step, kinds):
    rows = [r for r in json.load(open(run_json))["rows"] if r["step"] == step]
    return {k: float(np.median([r["k"] for r in rows if r["kind"] == k])) for k in kinds}


def progress_matched(base_run, arm_run, base_json, arm_json, grid):
    """取干预臂终端 loss, 在 base 的 ckpt 网格上找最接近的等 loss 点, 比较 per-kind k。
    **描述性**: 报差异大小, 不做因果判读修正 (spec P0-2)。"""
    l_arm = loss_at(arm_run, 30000)
    cand = [(s, loss_at(base_run, s)) for s in grid]
    cand = [(s, l) for s, l in cand if l is not None]
    s_m, l_m = min(cand, key=lambda t: abs(t[1] - l_arm))
    k_arm = k_at_step(arm_json, 30000, KINDS)
    k_base_term = k_at_step(base_json, 30000, KINDS)
    k_base_match = k_at_step(base_json, s_m, KINDS)
    return {"arm_loss_30k": l_arm, "base_matched_step": s_m, "base_matched_loss": l_m,
            "loss_gap_at_match": l_m - l_arm,
            "per_kind": {k: {"arm_30k": k_arm[k], "base_30k": k_base_term[k],
                             "base_matched": k_base_match[k],
                             "d_vs_base30k": k_arm[k] - k_base_term[k],
                             "d_vs_matched": k_arm[k] - k_base_match[k]} for k in KINDS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default="05_ivn_runs/ivn_prereg_readouts_v1.json")
    a = ap.parse_args()
    R = lambda arm: os.path.join(a.base, f"05_ivn_runs/ivn_{arm}")
    J = lambda arm: os.path.join(a.base, f"05_ivn_runs/ivn_{arm}/analysis_v2.json")

    res = {"schema": "ivn-prereg-readouts-v1",
           "spec": "IVN_spec_v2_260825 §1 强制记录的次级读数",
           "terminal_window": TERM,
           "exposure_ratio": {}, "align_injection": {}, "row_rescue": {},
           "progress_matched": {}}

    for arm in ARMS:
        print(f"[{arm}] exposure...", flush=True)
        res["exposure_ratio"][arm] = exposure(R(arm))
        print(f"[{arm}] align/inject...", flush=True)
        res["align_injection"][arm] = align_inject(R(arm))
        print(f"[{arm}] row rescue...", flush=True)
        res["row_rescue"][arm] = rescue(R(arm))

    grid = [0, 800, 3200, 5000, 10000, 15000, 20000, 25000, 30000]
    for arm in ["flatv", "nomom", "ropeperm"]:
        res["progress_matched"][arm] = progress_matched(R("base"), R(arm), J("base"), J(arm), grid)

    with open(os.path.join(a.base, a.out), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    # ---- 打印 ----
    print("\n=== A. exposure ratio (行级中位, 终端 30k) ===")
    for arm in ARMS:
        e = res["exposure_ratio"][arm]
        if e is None:
            print(f"{arm:9s} 未记录 (训练脚本仅 flatv 臂写入)")
        else:
            print(f"{arm:9s} " + " ".join(f"{k[:4]}={v['median']:.4f}"
                                          for k, v in sorted(e[30000].items())))
    print("\n=== C. row-normalization rescue k (终端窗中位) ===")
    print(f"{'arm':9s}" + "".join(f"{k[:4]:>9s}" for k in KINDS))
    for arm in ARMS:
        r = res["row_rescue"][arm]
        print(f"{arm:9s}" + "".join(f"{r[k]['k_before']:9.4f}" for k in KINDS) + "   before")
        print(f"{'':9s}" + "".join(f"{r[k]['k_after_row']:9.4f}" for k in KINDS) + "   after row-norm")
        print(f"{'':9s}" + "".join(f"{r[k]['k_after_col']:9.4f}" for k in KINDS) + "   after col-norm")
    print("\n=== D. progress-matched (描述性) ===")
    for arm, v in res["progress_matched"].items():
        print(f"{arm:9s} loss30k={v['arm_loss_30k']:.4f} → base 匹配步 {v['base_matched_step']} "
              f"(loss {v['base_matched_loss']:.4f}, 残差 {v['loss_gap_at_match']:+.4f})")
        for k in ["q_proj", "k_proj"]:
            p = v["per_kind"][k]
            print(f"          {k:8s} arm={p['arm_30k']:.4f} vs base30k={p['base_30k']:.4f} "
                  f"({p['d_vs_base30k']:+.4f}) vs matched={p['base_matched']:.4f} ({p['d_vs_matched']:+.4f})")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
