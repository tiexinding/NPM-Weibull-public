#!/usr/bin/env python
"""IVN 因果干预训练 (IVN_spec_v2_260825, fork of opt_train_uvar.py).

四臂干预:
  base     — contemporaneous control (标准 AdamW, 与 EA baseline 超参数逐位同)
  flatv    — u = m̂/(√v̄_row+ε), 消坐标级 v 异质, 保行级水平
  nomom    — β1=0, u = g/(√v̂+ε), 固定策略移除动量
  ropeperm — inv_freq 坐标映射固定置换 π, 同一组频率重排 m→坐标 指派

与 opt_train_uvar.py 的关键区别:
  - UVAR 是观测型 (变体只记录不回写); IVN 是干预型 (变体真实更新权重)
  - --arm 取代 --optim; 无 SGD 路径
  - flatv: opt.step() 后对分析矩阵 undo 标准更新, 重做 flat-v 更新
  - nomom: AdamW(betas=(0,β2))
  - ropeperm: 模型初始化后置换 inv_freq buffer

预算: 4 runs × 30k ≈ 0.28 卡·天。
自检: S1 解析验证 @t=250, S2 零梯度 decay, S3 ropeperm 双向 (preflight),
      S4 仪器锚 @t=250, S5 contemporaneous 锚 (终态).
"""
import os, math, json, time, hashlib, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
from scipy.stats import norm
from transformers import LlamaConfig, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["base", "flatv", "nomom", "ropeperm"])
ap.add_argument("--family", required=True, choices=["gaussian", "uniform", "laplace"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--horizon", type=int, default=30000)
ap.add_argument("--tokens", default="data/ea_tokens.npy")
ap.add_argument("--config", default="llama70m_config.json")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--warm", type=int, default=200)
ap.add_argument("--eta", type=float, default=1e-3)
ap.add_argument("--lwd", type=float, default=0.1)
ap.add_argument("--beta2", type=float, default=0.999)
ap.add_argument("--sigma", type=float, default=0.02)
ap.add_argument("--dryrun", action="store_true")
ap.add_argument("--ckpt_steps", default="0,800,3200,5000,10000,15000,20000,25000,30000")
a = ap.parse_args()

ARM = a.arm
os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/ckpt", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ETA, LWD = a.eta, a.lwd
B1 = 0.0 if ARM == "nomom" else 0.9       # nomom: β1=0 (K3: bc1=1, m̂=g)
B2, EPS = a.beta2, 1e-8
T, HOR, WARM, EMIN, BS, SEQ = a.steps, a.horizon, a.warm, 0.1, a.bs, a.seq
CKPT_STEPS = sorted(set(int(x) for x in a.ckpt_steps.split(",") if int(x) <= T or int(x) == 0))

# ropeperm π (K6: seed=3, 0 fixed points, Spearman ρ=0.0422)
PI = [20,17,5,14,28,27,4,15,2,18,16,1,23,26,7,13,12,29,6,22,9,11,31,19,21,0,8,30,3,25,24,10]

def lr(t):
    if t < WARM: return ETA * t / WARM
    f = (t - WARM) / (HOR - WARM)
    return ETA * EMIN + 0.5 * ETA * (1 - EMIN) * (1 + math.cos(math.pi * min(f, 1.0)))

# ---- S2 零梯度精确性检查 (承 UVAR; 对 decay 顺序独立校验) ----
def zero_grad_decay_check(lr_t):
    base_t = (torch.arange(16, dtype=torch.float64, device=dev).reshape(4, 4) + 1.0) * 0.013
    sign_t = torch.tensor([[1., -1., 1., -1.]] * 4, dtype=torch.float64, device=dev)
    p = torch.nn.Parameter(base_t * sign_t)
    W_pre = p.detach().clone()
    o = torch.optim.AdamW([p], lr=lr_t, weight_decay=LWD, betas=(B1, B2), eps=EPS)
    o.zero_grad(set_to_none=False); p.grad = torch.zeros_like(p)
    o.step()
    expect = W_pre * (1.0 - lr_t * LWD)
    dev_max = float((p.detach() - expect).abs().max())
    tol = 1e-12 * max(1.0, float(W_pre.abs().max()))
    ok = dev_max <= tol
    print(f"[IVN-{ARM}] S2 零梯度 decay 检 (β1={B1}, lr={lr_t:.3e}): "
          f"{'PASS' if ok else 'FAIL'} (max dev={dev_max:.3e}, tol={tol:.1e})", flush=True)
    return ok, dev_max

# ---- 模型 (torch seed → 三族逐位一致) ----
torch.manual_seed(a.seed)
cfg = LlamaConfig.from_json_file(a.config); model = LlamaForCausalLM(cfg).to(dev)

_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def is_analysis(n): return any(f"{k}.weight" in n for k in _KINDS)
ANA = sorted(n for n, _ in model.named_parameters() if is_analysis(n))
pmap = dict(model.named_parameters())
assert len(ANA) == cfg.num_hidden_layers * 7, f"分析矩阵数 {len(ANA)} ≠ {cfg.num_hidden_layers * 7}"

# ---- ropeperm: 置换 inv_freq ----
def apply_rope_perm(mdl, perm):
    """对 model.model.rotary_emb.inv_freq 应用置换 perm (transformers >=4.57 全局共享)."""
    perm_t = torch.tensor(perm, dtype=torch.long, device=dev) if not isinstance(perm, torch.Tensor) else perm
    rope = mdl.model.rotary_emb
    assert hasattr(rope, "inv_freq"), "rotary_emb missing inv_freq"
    orig = rope.inv_freq.clone()
    rope.inv_freq.copy_(orig[perm_t])
    if hasattr(rope, "original_inv_freq"):
        rope.original_inv_freq = rope.inv_freq.clone()
    return 1  # 全局共享, 只需改一处

# ---- 三族耦合 init (与 ea_train/opt_train 逐位同) ----
def family_tensor(U, family):
    if family == "gaussian": x = norm.ppf(U)
    elif family == "uniform": x = (2 * U - 1) * math.sqrt(3)
    else: h = U - 0.5; x = -np.sign(h) * np.log(1 - 2 * np.abs(h)) / math.sqrt(2)
    x = x - x.mean(); x = x / x.std() * a.sigma; return x

init_report = {}
with torch.no_grad():
    for idx, nm in enumerate(ANA):
        p = pmap[nm]; n = p.numel()
        rng_i = np.random.default_rng(np.random.SeedSequence([a.seed, 7700, idx]))
        U = np.clip(rng_i.random(n), 1e-7, 1 - 1e-7)
        x = family_tensor(U, a.family)
        g = family_tensor(U, "gaussian")
        sign_agree = float((np.sign(x) == np.sign(g)).mean())
        p.copy_(torch.tensor(x.reshape(p.shape), dtype=p.dtype, device=dev))
        init_report[nm] = {"sign_agree_vs_gaussian": round(sign_agree, 6),
                           "mean": float(x.mean()), "std": float(x.std())}
model.train()

# ---- 语料与批采样 ----
toks = np.load(a.tokens).astype(np.int64)
tok_meta = {"n_tok": int(len(toks)), "max_id": int(toks.max()),
            "sha256_16": hashlib.sha256(toks.tobytes()).hexdigest()[:16],
            "revisit_x": round(T * BS * SEQ / len(toks), 1)}
rng = np.random.default_rng(a.seed)
batch_hashes = []
def get_batch(log_hash=False):
    idx = rng.integers(0, len(toks) - SEQ - 1, BS)
    if log_hash: batch_hashes.append(hashlib.sha256(idx.tobytes()).hexdigest()[:12])
    return torch.tensor(np.stack([toks[i:i + SEQ] for i in idx]), dtype=torch.long, device=dev)

# ---- Preflight (probe batch + S2 + S3) ----
rng_probe = np.random.default_rng(np.random.SeedSequence([a.seed, 8800]))
pidx = rng_probe.integers(0, len(toks) - SEQ - 1, BS)
px = torch.tensor(np.stack([toks[i:i + SEQ] for i in pidx]), dtype=torch.long, device=dev)
out0 = model(input_ids=px, labels=px, output_hidden_states=True)
hs_rms = [float(h.detach().float().pow(2).mean().sqrt()) for h in out0.hidden_states]
step0_loss = float(out0.loss)
out0.loss.backward()
grad_rms = {nm: float(pmap[nm].grad.detach().float().pow(2).mean().sqrt())
            for nm in ANA if pmap[nm].grad is not None}
model.zero_grad(set_to_none=True)

# S3 for ropeperm: 双向检 (a) identity 等价 (b) 真 π 必须不同
s3_report = None
if ARM == "ropeperm":
    loss_identity = step0_loss
    logits_identity = out0.logits.detach()
    # (b) 应用真 π, forward 输出必须不同 (spec: "attention 输出必须不同")
    n_applied = apply_rope_perm(model, PI)
    out_perm = model(input_ids=px, labels=px)
    loss_perm = float(out_perm.loss)
    model.zero_grad(set_to_none=True)
    # 用 logits max-abs 差检 (loss 在 init 因聚合丢失差异, logits 保留逐位信号)
    s3_logit_maxdiff = float((logits_identity - out_perm.logits.detach()).abs().max())
    s3_loss_diff = abs(loss_perm - loss_identity)
    s3b_pass = s3_logit_maxdiff > 1e-3
    del logits_identity
    # 保存置换后的 inv_freq 作为记录
    inv_freq_after = model.model.rotary_emb.inv_freq.cpu().tolist()
    s3_report = {"loss_identity": loss_identity, "loss_perm": loss_perm,
                 "loss_diff": s3_loss_diff,
                 "logit_maxdiff": s3_logit_maxdiff,
                 "s3b_pass": s3b_pass,
                 "layers_applied": n_applied, "inv_freq_after_layer0": inv_freq_after}
    print(f"[IVN-ropeperm] S3: identity_loss={loss_identity:.6f} perm_loss={loss_perm:.6f} "
          f"loss_diff={s3_loss_diff:.6f} logit_maxdiff={s3_logit_maxdiff:.4f} "
          f"S3b={'PASS' if s3b_pass else 'FAIL'}", flush=True)
    if not s3b_pass:
        open(f"{a.out}/ABORT_S3_perm_no_effect", "w").close(); raise SystemExit(6)
    # π 保持生效, 训练全程使用
    # 强制释放 S3 中间张量 (两次 forward 占 ~20GB, 不清理则训练 OOM)
    del out_perm, out0
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"[IVN-ropeperm] S3 cleanup: VRAM freed", flush=True)

zg_ok, zg_dev = zero_grad_decay_check(lr(max(WARM, 1)))

preflight = {"arm": ARM, "family": a.family, "seed": a.seed, "beta1": B1,
             "step0_loss": step0_loss,
             "hidden_rms_per_layer": hs_rms,
             "grad_rms_median": float(np.median(list(grad_rms.values()))),
             "sign_agree_median": float(np.median([v["sign_agree_vs_gaussian"]
                                                   for v in init_report.values()])),
             "zero_grad_decay_check": {"pass": bool(zg_ok), "max_dev": zg_dev},
             "s3_ropeperm": s3_report,
             "tokens": tok_meta,
             "pi": PI if ARM == "ropeperm" else None}
json.dump({"preflight": preflight, "init_per_matrix": init_report},
          open(f"{a.out}/preflight.json", "w"), indent=1)
print(f"[IVN-{ARM}] preflight: loss={step0_loss:.4f} "
      f"sign_agree_med={preflight['sign_agree_median']:.4f} revisit={tok_meta['revisit_x']}x", flush=True)
if not zg_ok:
    open(f"{a.out}/ABORT_S2_zero_grad_decay", "w").close(); raise SystemExit(4)

# ---- Optimizer (AdamW; β1 由 ARM 决定) ----
opt = torch.optim.AdamW(model.parameters(), lr=ETA, weight_decay=LWD,
                         betas=(B1, B2), eps=EPS)

# ---- 能量累加器 (fp64) ----
acc = {nm: {q: {"row": torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev),
               "col": torch.zeros(pmap[nm].shape[1], dtype=torch.float64, device=dev)}
            for q in ("tot", "adp", "dec", "cross")} for nm in ANA}

# ---- mv 记录 (Ag=梯度径向, Au=更新径向, SF=符号翻转) ----
mv = {nm: {q: torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev)
           for q in ("Ag", "Au", "SF", "prev_sign")} for nm in ANA}

def save_ckpt(t):
    torch.save({nm: pmap[nm].detach().to(torch.float32).cpu() for nm in ANA},
               f"{a.out}/ckpt/step{t}.pt")
    # 能量
    dump = {}
    for nm in ANA:
        for q in ("tot", "adp", "dec", "cross"):
            for ax in ("row", "col"):
                dump[f"{nm}|{q}|{ax}"] = acc[nm][q][ax].cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/energy_step{t}.npz", **dump)
    # mv + optimizer state snapshot
    mvd = {}
    for nm in ANA:
        mvd[f"{nm}|Ag"] = mv[nm]["Ag"].cpu().numpy()
        mvd[f"{nm}|Au"] = mv[nm]["Au"].cpu().numpy()
        mvd[f"{nm}|SF"] = mv[nm]["SF"].cpu().numpy()
        st = opt.state.get(pmap[nm], {})
        m_ = st.get("exp_avg"); v_ = st.get("exp_avg_sq")
        if m_ is not None:
            w = pmap[nm].detach().double()
            mvd[f"{nm}|m_rad"] = (m_.double() * w).sum(1).cpu().numpy()
            mvd[f"{nm}|v_row"] = v_.double().sum(1).cpu().numpy()
            mvd[f"{nm}|m_row2"] = (m_.double() ** 2).sum(1).cpu().numpy()
            # exposure: flatv 臂记录 ||u_flat||/||u_adam|| per row (从同一 m/v state)
            if ARM == "flatv" and t > 0:
                sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                bc1c = 1 - B1 ** sv; bc2c = 1 - B2 ** sv
                mh = m_.double() / bc1c; vh = v_.double() / bc2c
                u_adam = mh / (vh.sqrt() + EPS)
                vbar = vh.mean(dim=1, keepdim=True)
                u_flat = mh / (vbar.sqrt() + EPS)
                r = u_flat.norm(dim=1) / (u_adam.norm(dim=1) + 1e-30)
                mvd[f"{nm}|exposure_ratio"] = r.cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/mv_step{t}.npz", **mvd)

save_ckpt(0)

# ---- 训练主循环 ----
loss200 = None; t0 = time.time(); s1_checked = False; id_checked = False
for t in range(1, T + 1):
    cur = lr(t)
    for g in opt.param_groups: g["lr"] = cur
    W_pre = {nm: pmap[nm].detach().clone() for nm in ANA}

    opt.zero_grad(set_to_none=True)
    x = get_batch(log_hash=(t <= 5))
    out = model(input_ids=x, labels=x); loss = out.loss
    loss.backward()

    # MVREC: raw 梯度径向 + 符号翻转
    with torch.no_grad():
        for nm in ANA:
            rg = (pmap[nm].grad.double() * W_pre[nm].double()).sum(1)
            mv[nm]["Ag"] += cur * rg
            sg = torch.sign(rg)
            mv[nm]["SF"] += ((sg != mv[nm]["prev_sign"]) & (mv[nm]["prev_sign"] != 0)).double()
            mv[nm]["prev_sign"] = sg

    # ---- 优化器步进 ----
    opt.step()

    # flatv: undo 标准 Adam 更新, 重做 flat-v 更新 (仅分析矩阵)
    # m/v state 在 opt.step() 中已正确更新 (只依赖梯度), 只改参数更新公式
    if ARM == "flatv":
        with torch.no_grad():
            for nm in ANA:
                st = opt.state[pmap[nm]]
                sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                bc1 = 1 - B1 ** sv; bc2 = 1 - B2 ** sv
                mhat = st["exp_avg"].double() / bc1
                vhat = st["exp_avg_sq"].double() / bc2
                vbar = vhat.mean(dim=1, keepdim=True)       # 行内摊平
                u_flat = mhat / (vbar.sqrt() + EPS)
                pmap[nm].copy_((W_pre[nm].double() * (1 - cur * LWD) - cur * u_flat).float())

    # ---- 能量分解 + Au 记录 ----
    ratios = []
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
            if t % 50 == 0:
                ratios.append(float(d_tot.norm() / (W_pre[nm].double().norm() + 1e-30)))
            au_inc = -(d_adp * W_pre[nm].double()).sum(1)
            mv[nm]["Au"] += au_inc

            # ---- S1/S4 解析自检 @t=250 ----
            if t == 250 and not s1_checked:
                st = opt.state.get(pmap[nm], {})
                if st.get("exp_avg") is not None:
                    sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                    bc1c = 1 - B1 ** sv; bc2c = 1 - B2 ** sv
                    mh = st["exp_avg"].double() / bc1c
                    vh = st["exp_avg_sq"].double() / bc2c
                    if ARM == "flatv":
                        vb = vh.mean(dim=1, keepdim=True)
                        u_check = mh / (vb.sqrt() + EPS)
                    else:   # base, nomom, ropeperm: standard u
                        u_check = mh / (vh.sqrt() + EPS)
                    au_state = cur * (u_check * W_pre[nm].double()).sum(1)
                    rel = float((au_state - au_inc).abs().max() / (au_inc.abs().max() + 1e-30))
                    if rel > 1e-4:
                        tag = f"S1_{ARM}" if ARM in ("flatv", "nomom") else "S4"
                        open(f"{a.out}/ABORT_{tag}_identity", "w").write(
                            f"{nm} rel={rel:.2e}\n")
                        raise SystemExit(5)
                    print(f"[IVN-{ARM}] S1/S4 PASS ({nm}: rel={rel:.2e})", flush=True)

    if t == 250: s1_checked = True

    # dryrun 分解恒等式 (残差定义恒真, 只校 fp64 累加精度)
    if a.dryrun and t == 3 and not id_checked:
        nm0 = ANA[0]; A0 = acc[nm0]
        lhs = A0["tot"]["row"]; rhs = A0["adp"]["row"] + A0["dec"]["row"] + 2 * A0["cross"]["row"]
        ok = torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-18)
        print(f"[IVN-{ARM}] 分解恒等式 @step3: {'PASS' if ok else 'FAIL'} "
              f"(max rel dev={float(((lhs - rhs).abs() / (lhs.abs() + 1e-30)).max()):.2e}; "
              f"注: 残差定义恒真, 仅校累加精度)", flush=True)
        id_checked = True

    # ---- 稳定性 gate (NaN / catastrophic spike) ----
    lv = float(loss)
    if not math.isfinite(lv):
        open(f"{a.out}/ABORT_nonfinite_step{t}", "w").close(); raise SystemExit(3)
    if t == 200: loss200 = lv
    if loss200 and t > 200 and lv > 10 * loss200:
        open(f"{a.out}/ABORT_catastrophic_step{t}", "w").close(); raise SystemExit(3)

    if t % 50 == 0:
        with open(f"{a.out}/loss.jsonl", "a") as f:
            f.write(json.dumps({"step": t, "loss": lv, "eta": cur,
                                "upd_w_ratio_med": float(np.median(ratios)) if ratios else None}) + "\n")
        print(f"[IVN-{ARM}] step{t} loss={lv:.4f} "
              f"u/w={np.median(ratios):.3e} ({(time.time() - t0) / t:.3f}s/step)", flush=True)

    if t in CKPT_STEPS:
        save_ckpt(t)

# ---- 终点 ----
if T >= HOR:
    torch.save(model.state_dict(), f"{a.out}/full_model_final.pt")

sum_lr = sum(lr(t) for t in range(1, T + 1))
meta = {"arm": ARM, "family": a.family, "seed": a.seed,
        "steps": T, "horizon": HOR, "warm": WARM, "emin": EMIN,
        "beta1": B1, "beta2": B2, "eta_peak": ETA, "lwd": LWD,
        "bs": BS, "seq": SEQ, "sigma": a.sigma,
        "sum_lr": sum_lr,
        "cum_decay_exposure": LWD * sum_lr,
        "sum_log1m_eta_lambda": sum(math.log1p(-lr(t) * LWD) for t in range(1, T + 1)),
        "zero_grad_decay_check": {"pass": bool(zg_ok), "max_dev": zg_dev},
        "s3_ropeperm": s3_report,
        "pi": PI if ARM == "ropeperm" else None,
        "batch_hashes_first5": batch_hashes,
        "wall_sec": time.time() - t0,
        "max_vram_gb": (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else None}
json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
print(f"[IVN-{ARM}] DONE: {T} steps, {meta['wall_sec']:.0f}s, "
      f"vram={meta['max_vram_gb']}, 累计衰减暴露={meta['cum_decay_exposure']:.3f}", flush=True)
