#!/usr/bin/env python3
"""IVN 补做 spec v2 预注册读数 J1 / J2(H^W) / J3 (离线, 现有 ckpt, 不需 GPU)。

背景: RESULTS_IVN_v1.md 只报了 Δ_spread(k), 未执行 spec v2 §2 预注册的 J1/J2 拆分/J3。
其中 ropeperm 用"全局 k 不变"判 O-mem 阴性, 与 spec line 117 直接冲突
("频率集合未改变, 故全局 k 不变完全可能, k 不变不可判为阴性") → 本脚本补 J3 判据。

口径 (与 t123_structure_v1.py / hw_weight_side_v1.py 对齐):
  X = 累计 median-centered Δlog(行 RMS), 区间 3200→30k (spec J1)
  pair_avg: X_pair[h,m] = (X[h,m] + X[h,m+32])/2   (旋转对二维不变量)
  J1  freq partial R² = (SS_res(layer+head) − SS_res(layer+head+freqbin)) / SS_res(layer+head)
  H^W = IQR(log10 行RMS²), 终端窗口 {20k,25k,30k} 中位
  J3  P_coord = ropeperm 32 维频率 profile 按物理坐标索引 (freq_profile 直接输出)
      P_freq[f] = P_coord[π⁻¹(f)]     ← 实现为 after[m]=before[π[m]], 故坐标 m 承载频率 π(m)
      C_freq = corr(P_freq, P_base);  C_coord = corr(P_coord, P_base);  ΔC = C_freq − C_coord

⚠ 口径偏差 (照报): IVN ckpt 网格 (3200,5000,10000,15000,20000,25000,30000) 比 T1 原网格
   (3200,10000,20000,25000,30000) 密, 累计 median-centered drift 对步网格有轻微依赖 →
   本脚本 J1 绝对值与 T1 基线 (q/k 0.28-0.39) 不做逐位对比, 只做臂间对比 (同网格可比)。
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

STEPS_J1 = [3200, 5000, 10000, 15000, 20000, 25000, 30000]   # spec: 区间 3200→30k
STEPS_TERM = [20000, 25000, 30000]                            # 终端窗口
H, DH, NF = 8, 64, 32        # llama70m_config.json: num_attention_heads=8, hidden=512 → 8×64; NF=DH/2
ARMS = ["base", "flatv", "nomom", "ropeperm"]
ROW_KINDS = ["q_proj", "k_proj", "v_proj"]    # 行轴带 head/freq 结构


def _load(ckpt_dir, step):
    return torch.load(os.path.join(ckpt_dir, f"step{step}.pt"),
                      map_location="cpu", weights_only=True)


def cum_rel_drift(ckpt_dir, steps):
    """{(kind, layer): X(512,)} — 累计 median-centered Δlog(行 RMS)。"""
    logr = []
    for s in steps:
        cur = {}
        for nm, tw in _load(ckpt_dir, s).items():
            proj = nm.rsplit(".", 2)[-2]
            if proj not in ROW_KINDS:
                continue
            layer = int(nm.split("layers.")[1].split(".")[0])
            w = tw.to(torch.float64)
            cur[(proj, layer)] = np.log(np.sqrt((w ** 2).mean(dim=1).numpy()))
        logr.append(cur)
    X = {}
    for key in logr[0]:
        d = np.zeros_like(logr[0][key])
        for a, b in zip(logr[:-1], logr[1:]):
            dd = b[key] - a[key]
            d += dd - np.median(dd)     # median-centering: 去全局尺度漂移, 留行间异质
        X[key] = d
    return X


def pair_avg(x):
    out = np.empty(H * NF)
    for h in range(H):
        blk = x[h * DH:(h + 1) * DH]
        out[h * NF:(h + 1) * NF] = (blk[:NF] + blk[NF:]) / 2.0
    return out


def freq_partial_r2(X_by_key, kind, freq_label=None):
    """OLS dummy: layer + head → +freqbin 的增量解释。
    freq_label 非空时按频率身份重标记 (ropeperm 频率跟随口径, spec K2(b)):
    坐标 m 承载的频率身份 = pi[m] (因 inv_freq_after[m]=inv_freq_before[pi[m]]),
    故此处传 pi (不是 pi 的逆; J3 的 profile 重排才用 pi_inv)。"""
    rows, ys = [], []
    for (kd, layer), x in X_by_key.items():
        if kd != kind:
            continue
        xp = pair_avg(x)
        for idx in range(len(xp)):
            h, m = idx // NF, idx % NF
            f = m if freq_label is None else int(np.asarray(freq_label)[m])
            rows.append((layer, h, f // 4))      # 8 freq bins
            ys.append(xp[idx])
    if not rows:
        return None
    rows = np.array(rows); y = np.array(ys); y = y - y.mean()

    def ssres(cols):
        mats = [np.ones((len(y), 1))]
        for ci in cols:
            vals = np.unique(rows[:, ci])
            mats.append((rows[:, ci][:, None] == vals[None, :]).astype(float))
        Xd = np.hstack(mats)
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        return float(((y - Xd @ beta) ** 2).sum())

    ss_lh, ss_lhf = ssres([0, 1]), ssres([0, 1, 2])
    return {"freq_partial_R2": (ss_lh - ss_lhf) / ss_lh,
            "R2_layer_head": 1 - ss_lh / float((y ** 2).sum())}


def freq_profile(X_by_key, kind, per_layer=False):
    """32 维频率 profile (物理坐标索引), 跨 layer×head 取中位。"""
    if per_layer:
        out = {}
        for (kd, layer), x in X_by_key.items():
            if kd != kind:
                continue
            xp = pair_avg(x)
            out[layer] = np.array([np.median([xp[h * NF + m] for h in range(H)])
                                   for m in range(NF)])
        return out
    allv = {m: [] for m in range(NF)}
    for (kd, layer), x in X_by_key.items():
        if kd != kind:
            continue
        xp = pair_avg(x)
        for idx in range(len(xp)):
            allv[idx % NF].append(xp[idx])
    return np.array([np.median(allv[m]) for m in range(NF)])


def hw_terminal(ckpt_dir):
    """H^W = IQR(log10 行RMS²); per kind: 层中位 → 终端窗口中位。"""
    per_step = []
    for s in STEPS_TERM:
        cur = {}
        for nm, tw in _load(ckpt_dir, s).items():
            proj = nm.rsplit(".", 2)[-2]
            if proj == "down_proj":       # 行轴不可比 (行=hidden 而非 head 结构), 与 HROW spec 一致
                continue
            layer = int(nm.split("layers.")[1].split(".")[0])
            w = tw.to(torch.float64)
            rms2 = (w ** 2).mean(dim=1).numpy()
            lg = np.log10(rms2)
            cur.setdefault(proj, []).append(float(np.percentile(lg, 75) - np.percentile(lg, 25)))
        per_step.append({k: float(np.median(v)) for k, v in cur.items()})
    kinds = per_step[0].keys()
    return {k: float(np.median([ps[k] for ps in per_step])) for k in kinds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default="05_ivn_runs/ivn_j1_j3_hw_v1.json")
    a = ap.parse_args()

    pi = json.load(open(os.path.join(a.base, "05_ivn_runs/ivn_ropeperm/run_meta.json")))["pi"]
    pi = np.array(pi, dtype=int)
    assert sorted(pi.tolist()) == list(range(NF)), "pi 非合法置换"
    pi_inv = np.empty(NF, dtype=int); pi_inv[pi] = np.arange(NF)   # π⁻¹
    # 自检: after[m] = before[pi[m]] (对账 run_meta 记录的 inv_freq)
    s3 = json.load(open(os.path.join(a.base, "05_ivn_runs/ivn_ropeperm/run_meta.json")))["s3_ropeperm"]
    before = 10000.0 ** (-np.arange(NF) / float(NF))
    dev = float(np.abs(before[pi] - np.array(s3["inv_freq_after_layer0"])).max())
    assert dev < 1e-6, f"pi 方向自检失败 dev={dev:.2e}"

    res = {"schema": "ivn-j1-j3-hw-v1",
           "spec": "IVN_spec_v2_260825 §2 (J1 / J2-H^W / J3)",
           "steps_J1": STEPS_J1, "steps_terminal": STEPS_TERM,
           "pi": pi.tolist(), "pi_direction": "inv_freq_after[m] = inv_freq_before[pi[m]]",
           "pi_selfcheck_maxdev": dev,
           "heads": H, "head_dim": DH, "n_freq_pairs": NF,
           "J1": {}, "H_W": {}, "J3": {}}

    X = {}
    for arm in ARMS:
        ck = os.path.join(a.base, f"05_ivn_runs/ivn_{arm}/ckpt")
        X[arm] = cum_rel_drift(ck, STEPS_J1)
        res["H_W"][arm] = hw_terminal(ck)
        res["J1"][arm] = {}
        for kd in ROW_KINDS:
            # ropeperm 额外报频率跟随口径 (spec K2(b))
            res["J1"][arm][kd] = {"coord_indexed": freq_partial_r2(X[arm], kd)}
            if arm == "ropeperm":
                res["J1"][arm][kd]["freq_indexed"] = freq_partial_r2(X[arm], kd, freq_label=pi)

    # ---- J3: 仅 ropeperm, 对 ivn_base profile ----
    for kd in ROW_KINDS:
        Pb = freq_profile(X["base"], kd)
        Pc = freq_profile(X["ropeperm"], kd)          # 物理坐标索引
        Pf = Pc[pi_inv]                                # P_freq[f] = P_coord[pi_inv[f]]
        c_f = float(np.corrcoef(Pf, Pb)[0, 1])
        c_c = float(np.corrcoef(Pc, Pb)[0, 1])
        # 逐层
        Lb, Lc = freq_profile(X["base"], kd, True), freq_profile(X["ropeperm"], kd, True)
        per_layer = {}
        for L in sorted(Lb):
            pf = float(np.corrcoef(Lc[L][pi_inv], Lb[L])[0, 1])
            pc = float(np.corrcoef(Lc[L], Lb[L])[0, 1])
            per_layer[L] = {"C_freq": pf, "C_coord": pc, "dC": pf - pc}
        # 置换零分布: 随机置换标签下 ΔC 的分布 (判 ΔC 是否超出偶然)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(2000):
            p = rng.permutation(NF)
            null.append(float(np.corrcoef(Pc[p], Pb)[0, 1]) - c_c)
        res["J3"][kd] = {"C_freq": c_f, "C_coord": c_c, "dC": c_f - c_c,
                         "per_layer": per_layer,
                         "n_layers_dC_pos": sum(1 for v in per_layer.values() if v["dC"] > 0),
                         "null_dC_p95": float(np.percentile(null, 95)),
                         "null_dC_mean": float(np.mean(null)),
                         "dC_exceeds_null_p95": bool((c_f - c_c) > np.percentile(null, 95)),
                         "is_negative_control": kd == "v_proj"}

    os.makedirs(os.path.dirname(os.path.join(a.base, a.out)), exist_ok=True)
    with open(os.path.join(a.base, a.out), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["J3"], indent=1)[:1500])
    print("\n--- J1 freq partial R² ---")
    for arm in ARMS:
        for kd in ROW_KINDS:
            e = res["J1"][arm][kd]
            s = f"{arm:9s} {kd:7s} coord={e['coord_indexed']['freq_partial_R2']:.4f}"
            if "freq_indexed" in e:
                s += f"  freq={e['freq_indexed']['freq_partial_R2']:.4f}"
            print(s)
    print("\n--- H^W (terminal) ---")
    for arm in ARMS:
        print(f"{arm:9s} " + " ".join(f"{k[:4]}={v:.4f}" for k, v in sorted(res["H_W"][arm].items())))
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
