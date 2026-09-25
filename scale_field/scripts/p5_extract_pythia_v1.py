#!/usr/bin/env python3
"""Paper#5 云端提取器 v1 (260903): 从 P4 尺寸序列 run 的 p5_ckpts/ (ana-only Pythia state_dict) 提取 P5 读数, 只留 json+npz 后可删 ckpt.

口径逐字复用 p5_kmix_on_ivn_v1.py / kmix_bridge_v1_1.py:
  khist: 1024-bin log10|w| 直方图, middle-80% (F∈[0.1,0.9]) 加权最小二乘 Weibull 拟合 → (k, R²_fit)
  rownorm: 行 RMS 归一并回缩到原总 RMS → k_norm
  hw_sd  = sd_i(log rowRMS_i)      (KMIX 桥统计量 H^W)
  hw_iqr = IQR_i(log10 rowRMS_i²)  (IVN 预登记读数 H^W_IQR, 二者不得互引)
  α_P: origin fit of Δ_k^mix = k_raw⁻² − k_norm⁻² on (H^W)² (n=层数, 单 run 单 step 小样本, 只作描述)
GPT-NeoX: fused qkv 按 per-head [q_h;k_h;v_h]×HD 切; rotary_pct=0.25 ⇒ 每 head 前 rot=HD/4 维做 RoPE, 频率对 (i, i+rot/2), NF=rot/2=8.
输出: <run>/p5_extract_v1.json (逐矩阵读数 + 每步 α_P + q/k 频率 profile) + <run>/p5_rowscale_v1.npz (每矩阵 log rowRMS / log colRMS 向量, 供离线重算).
"""
import argparse, json, os, re, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--delete_ckpts", action="store_true", help="提取成功且 assert 全过后删 p5_ckpts/*.pt")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
HB, HR = 1024, (-12.0, 2.0)
cfg = json.load(open(a.config))
HID, NH, NL = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
HD = HID // NH; ROT = int(HD * cfg.get("rotary_pct", 1.0)); NF = ROT // 2
KINDS = ("q", "k", "v", "o", "ffn_in", "ffn_out")


def fit_hist_r2(hist):
    hist = hist.astype(np.float64); total = hist.sum()
    edges = np.linspace(HR[0], HR[1], HB + 1)
    cdf_u = np.cumsum(hist) / total; cdf_l = (np.cumsum(hist) - hist) / total
    F = (cdf_u + cdf_l) / 2.0
    bc = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    m = (F >= 0.1) & (F <= 0.9) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc[m], np.ones(m.sum())]).T
    w = hist[m]; WX = X * w[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    res = y - X @ beta; ybar = (w * y).sum() / w.sum()
    r2f = 1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum()
    return float(beta[0]), float(r2f)


def khist(x):
    h = np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0]
    return fit_hist_r2(h)


def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True))
    z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())


def split_qkv(w):
    w3 = w.reshape(NH, 3, HD, HID)
    return {"q": w3[:, 0].reshape(NH * HD, HID), "k": w3[:, 1].reshape(NH * HD, HID), "v": w3[:, 2].reshape(NH * HD, HID)}


def matrices(sd):
    m = {}
    for b in range(NL):
        pre = f"gpt_neox.layers.{b}"
        for kk, w in split_qkv(sd[f"{pre}.attention.query_key_value.weight"]).items():
            m[(b, kk)] = w
        m[(b, "o")] = sd[f"{pre}.attention.dense.weight"]
        m[(b, "ffn_in")] = sd[f"{pre}.mlp.dense_h_to_4h.weight"]
        m[(b, "ffn_out")] = sd[f"{pre}.mlp.dense_4h_to_h.weight"]
    assert len(m) == NL * len(KINDS), len(m)
    return m


def origin_fit(rs):
    x = np.array([r["hw_sd"] ** 2 for r in rs]); y = np.array([r["k_raw"] ** -2 - r["k_norm"] ** -2 for r in rs])
    if len(rs) < 3 or (x @ x) == 0: return None
    al = float((x @ y) / (x @ x)); r0 = float(1 - ((y - al * x) ** 2).sum() / max((y ** 2).sum(), 1e-30))
    return {"alpha": al, "R0_sq": r0, "n": len(rs)}


def freq_profile(logr_rows):
    """q/k 行 log RMS (NH*HD,) → 每 head 前 ROT 维按频率对 (i, i+NF) 取均值 → (NH, NF)."""
    x = logr_rows.reshape(NH, HD)[:, :ROT]
    return (x[:, :NF] + x[:, NF:ROT]) / 2.0


# ---- 步列表: p5_ckpts + W_ckpts 终点 ----
pd_, wd_ = f"{a.run}/p5_ckpts", f"{a.run}/W_ckpts"
files = {}
for d in (pd_, wd_):
    if os.path.isdir(d):
        for f in os.listdir(d):
            mm = re.match(r"step(\d+)\.pt$", f)
            if mm: files[int(mm.group(1))] = f"{d}/{f}"
STEPS = sorted(files); assert len(STEPS) >= 2, f"need ≥2 ckpts, got {STEPS}"
print(f"[p5x] {os.path.basename(a.run)}: steps={STEPS} HID={HID} NH={NH} NL={NL} HD={HD} ROT={ROT} NF={NF}", flush=True)

rows, alpha, npz, fp = [], {}, {}, {}
t0 = time.time()
for t in STEPS:
    sd = torch.load(files[t], map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if k.startswith("gpt_neox.layers.")}
    M = matrices(sd); step_rows = []
    for (b, kd), w in M.items():
        w = w.double().numpy()
        kr, r2r = khist(w); kn, r2n = khist(rownorm(w))
        rr = np.sqrt((w ** 2).mean(1)); cr = np.sqrt((w ** 2).mean(0))
        rec = {"step": t, "L": b, "kind": kd, "shape": list(w.shape), "rms": float(np.sqrt((w ** 2).mean())),
               "k_raw": kr, "r2fit_raw": r2r, "k_norm": kn, "r2fit_norm": r2n,
               "hw_sd": float(np.log(rr).std()), "hw_iqr": float(np.subtract(*np.percentile(np.log10(rr ** 2), [75, 25]))),
               "hcol_sd": float(np.log(cr).std())}
        rows.append(rec); step_rows.append(rec)
        npz[f"s{t}_L{b}_{kd}_logrow"] = np.log(rr).astype(np.float32); npz[f"s{t}_L{b}_{kd}_logcol"] = np.log(cr).astype(np.float32)
        if kd in ("q", "k"): fp[(t, b, kd)] = freq_profile(np.log(rr))
    assert len(step_rows) == NL * len(KINDS)
    alpha[str(t)] = {fam: origin_fit([r for r in step_rows if r["kind"] in kinds])
                     for fam, kinds in {"q": ("q",), "k": ("k",), "qk_pooled": ("q", "k"), "v": ("v",), "ffn_in": ("ffn_in",)}.items()}
    print(f"[p5x] step{t}: {len(step_rows)} mats, qk α_P={alpha[str(t)]['qk_pooled']['alpha']:.3f} R0²={alpha[str(t)]['qk_pooled']['R0_sq']:.3f}, "
          f"min R²_fit raw={min(r['r2fit_raw'] for r in step_rows):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    del sd, M

# ---- q/k 频率 profile: 相邻步 Δlog r 按频率索引, 跨 layer×head 中位; 相邻区间符号相关 ----
prof = {}
for kd in ("q", "k"):
    prof[kd] = {}
    for i in range(1, len(STEPS)):
        t1, t0_ = STEPS[i], STEPS[i - 1]
        d = np.stack([fp[(t1, b, kd)] - fp[(t0_, b, kd)] for b in range(NL)])  # (NL, NH, NF)
        prof[kd][f"{t0_}-{t1}"] = np.median(d.reshape(-1, NF), axis=0).tolist()
    keys = list(prof[kd]); prof[kd]["adjacent_corr"] = {f"{keys[j-1]}|{keys[j]}": float(np.corrcoef(prof[kd][keys[j-1]], prof[kd][keys[j]])[0, 1])
                                                       for j in range(1, len(keys))} if len(keys) > 1 else {}
    for (t, b, k2), v in fp.items():
        if k2 == kd: npz[f"s{t}_L{b}_{kd}_freqprof"] = v.astype(np.float32)

r2min = min(r["r2fit_raw"] for r in rows); n_gate_fail = sum(r["r2fit_raw"] < 0.99 for r in rows)
out = {"schema": "p5-extract-pythia-v1", "run": os.path.basename(a.run), "config": os.path.basename(a.config),
       "dims": {"hidden": HID, "heads": NH, "layers": NL, "head_dim": HD, "rot": ROT, "nf": NF}, "steps": STEPS,
       "protocol": {"fit": "1024-bin log10|w| hist, middle-80% weighted LS Weibull", "hw_sd": "sd_i log rowRMS", "hw_iqr": "IQR_i log10 rowRMS^2",
                    "alpha": "origin fit (k_raw^-2 - k_norm^-2) ~ alpha*(hw_sd)^2 over layers; small-n descriptive only",
                    "freq_profile": "median over layers*heads of Δ(log rowRMS) per rotary frequency pair (i,i+NF), first ROT dims of each head"},
       "n_matrices": len(rows), "r2fit_raw_min": r2min, "n_r2fit_below_0.99": n_gate_fail,
       "alpha_P": alpha, "freq_profile": prof, "rows": rows}
json.dump(out, open(f"{a.run}/p5_extract_v1.json", "w"), indent=1)
np.savez_compressed(f"{a.run}/p5_rowscale_v1.npz", **npz)
assert len(rows) == len(STEPS) * NL * len(KINDS)
print(f"[p5x] DONE {out['run']}: {len(rows)} rows, R²_fit min {r2min:.4f} ({n_gate_fail} below 0.99), "
      f"json {os.path.getsize(a.run+'/p5_extract_v1.json')//1024}KB npz {os.path.getsize(a.run+'/p5_rowscale_v1.npz')//1024}KB", flush=True)
if a.delete_ckpts:
    for t, f in files.items():
        if f.startswith(pd_): os.remove(f)
    print(f"[p5x] deleted p5_ckpts of {out['run']} (W_ckpts 终点保留给 held-out eval)", flush=True)
