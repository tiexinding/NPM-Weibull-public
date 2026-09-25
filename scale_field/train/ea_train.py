#!/usr/bin/env python
"""E-A init-family intervention trainer (fork of three_force_spline.py, spec v2).

Fork 变更 (相对 P2 原版):
  - 剥离样条/三力机器; 保留离线构模/批采样/AdamW/cosine 日程骨架;
  - --horizon 与 --steps 分离 (评审#11 P0-2: 日程 horizon 恒 30000, --steps 只决定跑多远);
  - ① 三族耦合 init hook (秩次严格一致, 符号近似一致; step-0 报符号一致率);
  - ② step-0 pre-flight (probe batch: loss / 逐层 hidden RMS / 分析矩阵 grad RMS) — 仅诊断;
  - ③ E-A 15 点 ckpt 表 (分析矩阵 fp32; 终点另存全模型);
  - ④ 逐步四路能量 (total / decay 解析 / adaptive=差 / 交叉项; 行列两套, fp64 累加, ckpt 点转储);
  - ⑤ outcome-blind 在线门 (非有限值 / 灾难倍数; 族间分离永不作门);
  - --dryrun: 分解恒等式检查 + batch 耦合哈希 + 吞吐/显存实测。
"""
import os, math, json, time, glob, hashlib, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
from scipy.stats import norm
from transformers import LlamaConfig, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=["gaussian", "uniform", "laplace"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=30000, help="实际执行步数")
ap.add_argument("--horizon", type=int, default=30000, help="cosine 日程 horizon (恒 30000, 与 --steps 解耦)")
ap.add_argument("--tokens", default="data/ea_tokens.npy")
ap.add_argument("--config", default="llama70m_config.json")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--warm", type=int, default=200)
ap.add_argument("--eta", type=float, default=1e-3); ap.add_argument("--lwd", type=float, default=0.1)
ap.add_argument("--beta2", type=float, default=0.999)
ap.add_argument("--sigma", type=float, default=0.02)
ap.add_argument("--dryrun", action="store_true")
ap.add_argument("--ckpt_steps", default="0,50,100,200,400,800,1600,3200,5000,7500,10000,15000,20000,25000,30000")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/ckpt", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ETA, LWD, B1, B2, EPS = a.eta, a.lwd, 0.9, a.beta2, 1e-8
T, HOR, WARM, EMIN, BS, SEQ = a.steps, a.horizon, a.warm, 0.1, a.bs, a.seq
CKPT_STEPS = sorted(set(int(x) for x in a.ckpt_steps.split(",") if int(x) <= T or int(x) == 0))

def lr(t):
    if t < WARM: return ETA * t / WARM
    f = (t - WARM) / (HOR - WARM); return ETA * EMIN + 0.5 * ETA * (1 - EMIN) * (1 + math.cos(math.pi * min(f, 1.0)))

# ---- 模型 (torch seed → 非分析参数三族逐位一致) ----
torch.manual_seed(a.seed)
cfg = LlamaConfig.from_json_file(a.config); model = LlamaForCausalLM(cfg).to(dev)

_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def is_analysis(n): return any(f"{k}.weight" in n for k in _KINDS)
ANA = sorted(n for n, _ in model.named_parameters() if is_analysis(n))
pmap = dict(model.named_parameters())
assert len(ANA) == cfg.num_hidden_layers * 7, f"分析矩阵数 {len(ANA)} ≠ {cfg.num_hidden_layers*7}"

# ---- ① 三族耦合 init (U 与 batch rng 分离; 逆CDF; 精确零均值+σ) ----
def family_tensor(U, family):
    if family == "gaussian": x = norm.ppf(U)
    elif family == "uniform": x = (2 * U - 1) * math.sqrt(3)
    else: h = U - 0.5; x = -np.sign(h) * np.log(1 - 2 * np.abs(h)) / math.sqrt(2)
    x = x - x.mean(); x = x / x.std() * a.sigma
    return x

init_report = {}
with torch.no_grad():
    for idx, nm in enumerate(ANA):
        p = pmap[nm]; n = p.numel()
        rng_i = np.random.default_rng(np.random.SeedSequence([a.seed, 7700, idx]))
        U = np.clip(rng_i.random(n), 1e-7, 1 - 1e-7)
        x = family_tensor(U, a.family)
        g = family_tensor(U, "gaussian")           # 符号一致率参照 (同 U 的高斯臂)
        sign_agree = float((np.sign(x) == np.sign(g)).mean())
        p.copy_(torch.tensor(x.reshape(p.shape), dtype=p.dtype, device=dev))
        init_report[nm] = {"sign_agree_vs_gaussian": round(sign_agree, 6),
                           "mean": float(x.mean()), "std": float(x.std())}
model.train()

# ---- 语料与批采样 (原版同构; rng 只喂 batch) ----
toks = np.load(a.tokens).astype(np.int64)
tok_meta = {"n_tok": int(len(toks)), "max_id": int(toks.max()),
            "sha256_16": hashlib.sha256(toks.tobytes()).hexdigest()[:16],
            "revisit_x": round(T * BS * SEQ / len(toks), 1)}
rng = np.random.default_rng(a.seed)
batch_hashes = []
def batch(log_hash=False):
    idx = rng.integers(0, len(toks) - SEQ - 1, BS)
    if log_hash: batch_hashes.append(hashlib.sha256(idx.tobytes()).hexdigest()[:12])
    return torch.tensor(np.stack([toks[i:i + SEQ] for i in idx]), dtype=torch.long, device=dev)

# ---- ② pre-flight (probe batch 独立 rng; 只诊断不作门; 在 optimizer 创建前) ----
rng_probe = np.random.default_rng(np.random.SeedSequence([a.seed, 8800]))
pidx = rng_probe.integers(0, len(toks) - SEQ - 1, BS)
px = torch.tensor(np.stack([toks[i:i + SEQ] for i in pidx]), dtype=torch.long, device=dev)
out0 = model(input_ids=px, labels=px, output_hidden_states=True)
hs_rms = [float(h.detach().float().pow(2).mean().sqrt()) for h in out0.hidden_states]
out0.loss.backward()
grad_rms = {nm: float(pmap[nm].grad.detach().float().pow(2).mean().sqrt()) for nm in ANA if pmap[nm].grad is not None}
model.zero_grad(set_to_none=True)
preflight = {"family": a.family, "seed": a.seed, "step0_loss": float(out0.loss),
             "hidden_rms_per_layer": hs_rms,
             "grad_rms_median": float(np.median(list(grad_rms.values()))),
             "sign_agree_median": float(np.median([v["sign_agree_vs_gaussian"] for v in init_report.values()])),
             "tokens": tok_meta}
json.dump({"preflight": preflight, "init_per_matrix": init_report},
          open(f"{a.out}/preflight.json", "w"), indent=1)
print(f"[EA] {a.family} s{a.seed} step0: loss={preflight['step0_loss']:.4f} "
      f"sign_agree_med={preflight['sign_agree_median']:.4f} revisit={tok_meta['revisit_x']}x", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=ETA, weight_decay=LWD, betas=(B1, B2), eps=EPS)

# ---- ④ 能量累加器 (fp64) ----
acc = {nm: {q: {"row": torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev),
               "col": torch.zeros(pmap[nm].shape[1], dtype=torch.float64, device=dev)}
            for q in ("tot", "adp", "dec", "cross")} for nm in ANA}

def save_ckpt(t):
    torch.save({nm: pmap[nm].detach().to(torch.float32).cpu() for nm in ANA}, f"{a.out}/ckpt/step{t}.pt")
    dump = {}
    for nm in ANA:
        for q in ("tot", "adp", "dec", "cross"):
            for ax in ("row", "col"):
                dump[f"{nm}|{q}|{ax}"] = acc[nm][q][ax].cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/energy_step{t}.npz", **dump)

save_ckpt(0)
loss200 = None; t0 = time.time(); id_checked = False
for t in range(1, T + 1):
    cur = lr(t)
    for g in opt.param_groups: g["lr"] = cur
    W_pre = {nm: pmap[nm].detach().clone() for nm in ANA}
    opt.zero_grad(set_to_none=True)
    x = batch(log_hash=(t <= 5))
    out = model(input_ids=x, labels=x); loss = out.loss
    loss.backward(); opt.step()
    with torch.no_grad():
        for nm in ANA:
            d_tot = (pmap[nm].detach() - W_pre[nm]).double()
            d_dec = (-cur * LWD) * W_pre[nm].double()
            d_adp = d_tot - d_dec
            A = acc[nm]
            A["tot"]["row"] += (d_tot ** 2).sum(1); A["tot"]["col"] += (d_tot ** 2).sum(0)
            A["adp"]["row"] += (d_adp ** 2).sum(1); A["adp"]["col"] += (d_adp ** 2).sum(0)
            A["dec"]["row"] += (d_dec ** 2).sum(1); A["dec"]["col"] += (d_dec ** 2).sum(0)
            A["cross"]["row"] += (d_adp * d_dec).sum(1); A["cross"]["col"] += (d_adp * d_dec).sum(0)
    if a.dryrun and t == 3 and not id_checked:   # 分解恒等式: E_tot = E_adp + E_dec + 2·cross
        nm = ANA[0]; A = acc[nm]
        lhs = A["tot"]["row"]; rhs = A["adp"]["row"] + A["dec"]["row"] + 2 * A["cross"]["row"]
        ok = torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-18)
        print(f"[EA] 分解恒等式 @step3: {'PASS' if ok else 'FAIL'} "
              f"(max rel dev={float(((lhs-rhs).abs()/(lhs.abs()+1e-30)).max()):.2e})", flush=True)
        id_checked = True
    # ---- ⑤ outcome-blind 门 ----
    lv = float(loss)
    if not math.isfinite(lv):
        open(f"{a.out}/ABORT_nonfinite_step{t}", "w").close(); raise SystemExit(3)
    if t == 200: loss200 = lv
    if loss200 and t > 200 and lv > 10 * loss200:
        open(f"{a.out}/ABORT_catastrophic_step{t}", "w").close(); raise SystemExit(3)
    if t % 50 == 0:
        with open(f"{a.out}/loss.jsonl", "a") as f:
            f.write(json.dumps({"step": t, "loss": lv, "eta": cur}) + "\n")
        print(f"[EA] {a.family} s{a.seed} step{t} loss={lv:.4f} ({(time.time()-t0)/t:.3f}s/step)", flush=True)
    if t in CKPT_STEPS:
        save_ckpt(t)

if T >= HOR:   # 终点存全模型
    torch.save(model.state_dict(), f"{a.out}/full_model_final.pt")
meta = {"family": a.family, "seed": a.seed, "steps": T, "horizon": HOR, "warm": WARM, "emin": EMIN,
        "eta": ETA, "lwd": LWD, "beta2": B2, "bs": BS, "seq": SEQ, "sigma": a.sigma,
        "cum_decay_exposure": LWD * sum(lr(t) for t in range(1, T + 1)),
        "batch_hashes_first5": batch_hashes,
        "wall_sec": time.time() - t0,
        "max_vram_gb": (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else None}
json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
print(f"[EA] DONE {a.family} s{a.seed}: {T} steps, {meta['wall_sec']:.0f}s, "
      f"vram={meta['max_vram_gb']}, 累计衰减暴露={meta['cum_decay_exposure']:.3f}", flush=True)
