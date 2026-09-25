#!/usr/bin/env python
"""TCLOCK-D trainer (fork of opt_train_uvar.py; spec TCLOCK_pilot_spec_v1_260826 v1.1).

相对 opt_train_uvar.py 的变更 (只列差异):
  1. 语料: --arm {D1,D2,D3,D4} × --seed → 进程内确定性构造 f-shuffle+tile 流
     (partial_shuffle/tile_to 逐字复制自 P4 p4_build_grid_corpora.py; 母序列 = ea_tokens.npy
      前缀 [wt103 全量, Pythia tokenizer, 与 E-A 底座 canonical 语料同源]
      ⚠️工程决定: spec 写 wikitext_30M.npy, 但 bs24×seq512×5000 步=61.4M tokens > 30M,
      r01 的 U=T 放不下 → 母序列升级为同管线同 tokenizer 的 wt103 全量; 记入 manifest 待 Go/No-Go);
     🔴 f 语义: f=token 打乱比例, f000=结构完整, f100=全打乱 (源: P4 语料设计方案 §二 旋钮 A)。
  2. 批采样: 随机采样 → **顺序遍历流** (batch t = 流的第 t 个连续 BS×SEQ 块), 四臂 batch
     index schedule 恒等 (paired 判据 §5.3); rng 不再喂 batch ⇒ init 与语料完全解耦。
  3. 记录: 轻量 snapshot @SNAP_STEPS (行级统计 + 1024-bin 直方图, 不存整模型);
     完整 ckpt (model+optimizer+RNG) @CKPT_FULL; 逐步 loss/u-w ratio/能量与 mv 累加照旧。
  4. held-out: 每臂按 f 做 structure-matched held (母序列尾部 tail, 同 f shuffle, 与训练区零重叠),
     eval @ {400,800,1600,3200,5000}; block-hash 无泄漏检查。
  5. 移除: SGD 分支 / UVAR 四对照 (Aflatv/Apermv/Anomom) — 首批不判优化器, 省每步开销。
     保留: Ag/Au/SF/Atrue、能量 tot/adp/dec/cross、零梯度 decay 检查、outcome-blind 门。
用法: python tclock_train.py --arm D1 --seed 1 --out runs/tclock_D1_s1 [--steps 5000] [--dryrun]
"""
import os, math, json, time, hashlib, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
from scipy.stats import norm
from transformers import LlamaConfig, LlamaForCausalLM

ARMS = {"D1": (0.00, 1), "D2": (1.00, 1), "D3": (1.00, 64), "D4": (0.00, 64), "D5": (0.00, 4), "D6": (0.00, 16)}  # D5/D6: 8-26 重复剂量补充 (f=0, ρ=4/16)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=sorted(ARMS))
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--horizon", type=int, default=30000, help="cosine horizon 与 E-A 一致, 不随 steps 缩")
ap.add_argument("--tokens", default="data/ea_tokens.npy")
ap.add_argument("--config", default="llama70m_config.json")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--warm", type=int, default=200)
ap.add_argument("--eta", type=float, default=1e-3)
ap.add_argument("--lwd", type=float, default=0.1)
ap.add_argument("--sigma", type=float, default=0.02)
ap.add_argument("--held_n", type=int, default=524288, help="held-out token 数 (0.5M)")
ap.add_argument("--dryrun", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--extend_base", type=int, default=0, help="扩展模式: 原始步数 T0 (前缀语料按 T0 构造以保 CRN, T0 后接母序列 held 区之后的新鲜 token [r01] 或继续平铺 [r>1])")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/ckpt", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ETA, LWD, B1, B2, EPS = a.eta, a.lwd, 0.9, 0.999, 1e-8
T, HOR, WARM, EMIN, BS, SEQ = a.steps, a.horizon, a.warm, 0.1, a.bs, a.seq
F_SH, RHO = ARMS[a.arm]
SNAP_STEPS = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400,
              1600, 2000, 2400, 2800, 3200, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 9000, 10000]
CKPT_FULL = [0, 200, 400, 800, 1600, 3200, 5000, 6500, 8000, 10000]
HELD_EVAL = [400, 800, 1600, 3200, 5000, 6500, 8000, 10000]
SNAP_STEPS = sorted(set([s for s in SNAP_STEPS if s <= T] + [T]))
CKPT_FULL = sorted(set([s for s in CKPT_FULL if s <= T] + [T]))
HELD_EVAL = [s for s in HELD_EVAL if s <= T]

def lr(t):
    if t < WARM: return ETA * t / WARM
    f = (t - WARM) / (HOR - WARM); return ETA * EMIN + 0.5 * ETA * (1 - EMIN) * (1 + math.cos(math.pi * min(f, 1.0)))

# ---- 语料构造 (partial_shuffle/tile_to 逐字复制自 p4_build_grid_corpora.py) ----
def partial_shuffle(block, f, rng):
    if f <= 0: return block
    n = len(block); k = int(round(f * n))
    if k <= 1: return block
    pos = rng.choice(n, size=k, replace=False)
    vals = block[pos].copy(); rng.shuffle(vals); block[pos] = vals
    return block

def tile_to(block, total):
    reps = int(np.ceil(total / len(block)))
    return np.tile(block, reps)[:total]

full = np.load(a.tokens)
T0 = a.extend_base if a.extend_base else T          # 语料构造基准步数 (扩展模式下=原始 T, 保 CRN)
T0_TOK = T0 * BS * SEQ
T_TOK = T * BS * SEQ
U = T0_TOK // RHO
EXT_TOK = T_TOK - T0_TOK
assert len(full) >= T0_TOK + a.held_n + max(EXT_TOK, 0) + SEQ, \
    f"母序列不足: {len(full)} < {T0_TOK + a.held_n + EXT_TOK} (r01 扩展上限 ≈ {(len(full)-a.held_n)//(BS*SEQ)} 步)"
rng_sh = np.random.default_rng(np.random.SeedSequence([9900, a.seed, int(F_SH * 100)]))
blk = partial_shuffle(np.asarray(full[:U]).astype(np.int64).copy(), F_SH, rng_sh)
stream = tile_to(blk, T0_TOK)
if EXT_TOK > 0:
    if RHO == 1:   # r01: held 区之后的新鲜 token, 同 f 变换 (唯一性与零泄漏保持)
        src = np.asarray(full[T0_TOK + a.held_n: T0_TOK + a.held_n + EXT_TOK]).astype(np.int64).copy()
        rng_e = np.random.default_rng(np.random.SeedSequence([9922, a.seed, int(F_SH * 100)]))
        ext = partial_shuffle(src, F_SH, rng_e)
    else:          # r>1: 继续平铺同一 U-块 (重复语义自然延续)
        ext = tile_to(blk, T0_TOK + EXT_TOK)[T0_TOK:]
    stream = np.concatenate([stream, ext])
# held-out: 母序列尾部 (训练前缀之外), 同 f 变换 (structure-matched), 独立 rng
tail = np.asarray(full[T0_TOK: T0_TOK + a.held_n]).astype(np.int64).copy()
rng_h = np.random.default_rng(np.random.SeedSequence([9911, a.seed, int(F_SH * 100)]))  # held 与原 run 逐位一致 (T0 基准)
held = partial_shuffle(tail, F_SH, rng_h)
# 无泄漏检查: 8-gram 块 hash 交集 (train 唯一块 vs held)
def block_hashes(x, k=1024, n=2000, rng=None):
    if len(x) <= k + 1:
        return {hashlib.sha256(x.tobytes()).hexdigest()}
    idx = (rng or np.random.default_rng(0)).integers(0, len(x) - k, n)
    return {hashlib.sha256(x[i:i + k].tobytes()).hexdigest() for i in idx}
h_tr = block_hashes(blk, rng=np.random.default_rng(1))
h_he = block_hashes(held, rng=np.random.default_rng(2))
leak = len(h_tr & h_he)
corpus_meta = {"arm": a.arm, "f_shuffle": F_SH, "rho": RHO, "U": int(U), "T_tok": int(T_TOK),
               "unique_frac_measured": float(len(np.unique(blk)) / len(blk)) if U <= 2_000_000 else None,
               "stream_sha256_16": hashlib.sha256(stream.tobytes()).hexdigest()[:16],
               "held_sha256_16": hashlib.sha256(held.tobytes()).hexdigest()[:16],
               "mother_sha256_16": hashlib.sha256(np.asarray(full[:1_000_000]).tobytes()).hexdigest()[:16],
               "leak_blocks_1024": int(leak),
               "f_semantics": "f=token 打乱比例; f000=结构完整/可预测性高, f100=全打乱"}
if leak > 0:
    open(f"{a.out}/ABORT_heldout_leak", "w").write(str(leak)); raise SystemExit(6)
# 实测打乱比例 (f100 应 ~1-1/n 位置变值; f000 应 0)
probe = min(U, 1_000_000)
moved = float((blk[:probe] != np.asarray(full[:probe]).astype(np.int64)).mean())
corpus_meta["measured_moved_frac"] = moved
if U >= 10_000:   # 微型 dry-run 配置下小 U 的 moved 统计无意义
    assert (F_SH == 0 and moved == 0.0) or (F_SH == 1.0 and moved > 0.9), f"打乱比例实测异常 {moved}"

# ---- 模型 init (gaussian, paired: 只依赖 seed, 与臂无关) ----
torch.manual_seed(a.seed)
cfg = LlamaConfig.from_json_file(a.config); model = LlamaForCausalLM(cfg).to(dev)
_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def is_analysis(n): return any(f"{k}.weight" in n for k in _KINDS)
ANA = sorted(n for n, _ in model.named_parameters() if is_analysis(n))
pmap = dict(model.named_parameters())
assert len(ANA) == cfg.num_hidden_layers * 7
with torch.no_grad():
    for idx, nm in enumerate(ANA):
        p = pmap[nm]; n = p.numel()
        rng_i = np.random.default_rng(np.random.SeedSequence([a.seed, 7700, idx]))
        Uu = np.clip(rng_i.random(n), 1e-7, 1 - 1e-7)
        x = norm.ppf(Uu); x = x - x.mean(); x = x / x.std() * a.sigma
        p.copy_(torch.tensor(x.reshape(p.shape), dtype=p.dtype, device=dev))
init_sha = hashlib.sha256(b"".join(pmap[nm].detach().cpu().numpy().tobytes() for nm in ANA[:3])).hexdigest()[:16]
model.train()

# ---- 顺序批 (paired: index schedule 与臂无关) ----
def batch(t):
    o = (t - 1) * BS * SEQ
    x = stream[o:o + BS * SEQ].reshape(BS, SEQ)
    return torch.tensor(x, dtype=torch.long, device=dev)
sched_hash = hashlib.sha256(f"seq:{BS}x{SEQ}x{T}".encode()).hexdigest()[:12]

held_batches = [torch.tensor(held[i:i + BS * SEQ].reshape(BS, SEQ), dtype=torch.long, device=dev)
                for i in range(0, min(len(held), 20 * BS * SEQ), BS * SEQ)][:20]
def held_eval():
    model.eval(); tot = 0.0
    with torch.no_grad():
        for hb in held_batches:
            tot += float(model(input_ids=hb, labels=hb).loss)
    model.train(); return tot / len(held_batches)

# ---- 零梯度 decay 检查 (同 uvar §6.9) ----
def zero_grad_decay_check(lr_t):
    base = (torch.arange(16, dtype=torch.float64, device=dev).reshape(4, 4) + 1.0) * 0.013
    sign = torch.tensor([[1., -1., 1., -1.]] * 4, dtype=torch.float64, device=dev)
    p = torch.nn.Parameter(base * sign); W_pre = p.detach().clone()
    o = torch.optim.AdamW([p], lr=lr_t, weight_decay=LWD, betas=(B1, B2), eps=EPS)
    o.zero_grad(set_to_none=False); p.grad = torch.zeros_like(p); o.step()
    expect = W_pre * (1.0 - lr_t * LWD)
    dv = float((p.detach() - expect).abs().max()); ok = dv <= 1e-12 * max(1.0, float(W_pre.abs().max()))
    print(f"[TCLOCK] 零梯度检查: {'PASS' if ok else 'FAIL'} (dev={dv:.2e})", flush=True)
    return ok, dv
zg_ok, zg_dev = zero_grad_decay_check(lr(max(WARM, 1)))
if not zg_ok:
    open(f"{a.out}/ABORT_zero_grad_decay_check", "w").close(); raise SystemExit(4)

opt = torch.optim.AdamW(model.parameters(), lr=ETA, weight_decay=LWD, betas=(B1, B2), eps=EPS)

acc = {nm: {q: {"row": torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev),
                "col": torch.zeros(pmap[nm].shape[1], dtype=torch.float64, device=dev)}
            for q in ("tot", "adp", "dec", "cross")} for nm in ANA}
mv = {nm: {q: torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev)
           for q in ("Ag", "Au", "SF", "prev_sign", "Atrue")} for nm in ANA}
upd_row = {nm: torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev) for nm in ANA}  # 累计行级更新范数²

HIST_BINS, HIST_RANGE = 1024, (-12.0, 2.0)
def snap_light(t):
    d = {}
    for nm in ANA:
        w = pmap[nm].detach().double()
        d[f"{nm}|logr"] = (0.5 * torch.log((w ** 2).mean(1))).cpu().numpy().astype(np.float32)
        hist = np.histogram(np.log10(np.maximum(np.abs(w.cpu().numpy().ravel()), 1e-13)),
                            bins=HIST_BINS, range=HIST_RANGE)[0].astype(np.int32)
        d[f"{nm}|hist1024"] = hist
        st = opt.state.get(pmap[nm], {})
        if st.get("exp_avg") is not None:
            d[f"{nm}|m_row2"] = (st["exp_avg"].double() ** 2).sum(1).cpu().numpy()
            d[f"{nm}|v_row"] = st["exp_avg_sq"].double().sum(1).cpu().numpy()
        for q in ("Ag", "Au", "Atrue", "SF"):
            d[f"{nm}|{q}"] = mv[nm][q].cpu().numpy()
        d[f"{nm}|upd_row2"] = upd_row[nm].cpu().numpy()
        for q in ("tot", "adp", "dec", "cross"):
            d[f"{nm}|E_{q}_row"] = acc[nm][q]["row"].cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/snap_step{t}.npz", **d)

def ckpt_full(t):
    """两级存储 (dry-run 存储外推修正: full 含 opt state 811MB, 7点×12 run 超盘):
    - 权重级 step{t}.pt: 仅分析矩阵 fp32 (~44MB), 供行归一救援/复核, 7 时点全保留;
    - 滚动 full_latest.pt: model+opt+rng+累加器, 原子写只留最新, 供断点续跑。"""
    torch.save({nm: pmap[nm].detach().to(torch.float32).cpu() for nm in ANA},
               f"{a.out}/ckpt/step{t}.pt")
    tmp = f"{a.out}/ckpt/.full_latest.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state() if dev == "cuda" else None),
                "acc": {nm: {q: {ax: acc[nm][q][ax].cpu() for ax in ("row", "col")}
                             for q in acc[nm]} for nm in ANA},
                "mv": {nm: {q: mv[nm][q].cpu() for q in mv[nm]} for nm in ANA},
                "upd_row": {nm: upd_row[nm].cpu() for nm in ANA},
                "loss200": globals().get("loss200"),
                "step": t}, tmp)
    os.replace(tmp, f"{a.out}/ckpt/full_latest.pt")

json.dump({"corpus": corpus_meta, "init_sha_16": init_sha, "sched_hash": sched_hash,
           "zero_grad_decay_check": {"pass": bool(zg_ok), "dev": zg_dev},
           "snap_steps": SNAP_STEPS, "ckpt_full": CKPT_FULL, "held_eval": HELD_EVAL},
          open(f"{a.out}/preflight.json", "w"), indent=1)
START = 1
if a.resume:
    import glob as _g
    lp = f"{a.out}/ckpt/full_latest.pt"
    assert os.path.exists(lp), "--resume 但无 full_latest.pt"
    sd = torch.load(lp, map_location=dev, weights_only=False)
    rs = sd["step"]
    model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
    torch.set_rng_state(sd["torch_rng"].cpu())
    if dev == "cuda" and sd["cuda_rng"] is not None:
        torch.cuda.set_rng_state(sd["cuda_rng"].cpu())
    for nm in ANA:
        for q in acc[nm]:
            for ax in ("row", "col"):
                acc[nm][q][ax] = sd["acc"][nm][q][ax].to(dev)
        for q in mv[nm]:
            mv[nm][q] = sd["mv"][nm][q].to(dev)
        upd_row[nm] = sd["upd_row"][nm].to(dev)
    START = rs + 1
    loss200 = sd.get("loss200")
    print(f"[TCLOCK] resume from full_step{rs}", flush=True)
else:
    snap_light(0); ckpt_full(0)
print(f"[TCLOCK] {a.arm} s{a.seed}: f={F_SH} ρ={RHO} U={U:,} stream={corpus_meta['stream_sha256_16']} "
      f"init={init_sha} leak=0", flush=True)

loss200 = globals().get("loss200"); t0 = time.time()
for t in range(START, T + 1):
    cur = lr(t)
    for g in opt.param_groups: g["lr"] = cur
    W_pre = {nm: pmap[nm].detach().clone() for nm in ANA}
    opt.zero_grad(set_to_none=True)
    x = batch(t)
    out = model(input_ids=x, labels=x); loss = out.loss
    loss.backward()
    with torch.no_grad():
        for nm in ANA:
            rg = (pmap[nm].grad.double() * W_pre[nm].double()).sum(1)
            mv[nm]["Ag"] += cur * rg
            sg = torch.sign(rg)
            mv[nm]["SF"] += ((sg != mv[nm]["prev_sign"]) & (mv[nm]["prev_sign"] != 0)).double()
            mv[nm]["prev_sign"] = sg
    opt.step()
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
            upd_row[nm] += (d_tot ** 2).sum(1)
            mv[nm]["Au"] += -(d_adp * W_pre[nm].double()).sum(1)
            st = opt.state.get(pmap[nm], {})
            if st.get("exp_avg") is not None:
                bc1 = 1 - B1 ** st["step"]; bc2 = 1 - B2 ** st["step"]
                u_t = (st["exp_avg"] / bc1) / ((st["exp_avg_sq"] / bc2).sqrt() + EPS)
                mv[nm]["Atrue"] += cur * (u_t.double() * W_pre[nm].double()).sum(1)
            if t % 50 == 0:
                ratios.append(float(d_tot.norm() / (W_pre[nm].double().norm() + 1e-30)))
    lv = float(loss.detach())
    if not math.isfinite(lv):
        open(f"{a.out}/ABORT_nonfinite_step{t}", "w").close(); raise SystemExit(3)
    if t == 200: loss200 = lv
    if loss200 and t > 200 and lv > 10 * loss200:
        open(f"{a.out}/ABORT_catastrophic_step{t}", "w").close(); raise SystemExit(3)
    if t % 50 == 0:
        with open(f"{a.out}/loss.jsonl", "a") as f:
            f.write(json.dumps({"step": t, "loss": lv, "eta": cur,
                                "upd_w_ratio_med": float(np.median(ratios)) if ratios else None}) + "\n")
        print(f"[TCLOCK] {a.arm} s{a.seed} step{t} loss={lv:.4f} ({(time.time()-t0)/t:.3f}s/step)", flush=True)
    if t in HELD_EVAL:
        hl = held_eval()
        with open(f"{a.out}/held.jsonl", "a") as f:
            f.write(json.dumps({"step": t, "held_loss": hl, "train_loss": lv, "gap": lv - hl}) + "\n")
    if t in SNAP_STEPS:
        snap_light(t)
    if t in CKPT_FULL:
        ckpt_full(t)

if a.dryrun:   # dry-run 覆盖 held 路径 (正式 HELD_EVAL 首点 400 > dry 步数)
    hl = held_eval()
    print(f"[TCLOCK] dryrun held_eval: {hl:.4f}", flush=True)

meta = {"arm": a.arm, "seed": a.seed, "steps": T, "horizon": HOR, "warm": WARM,
        "eta": ETA, "lwd": LWD, "beta1": B1, "beta2": B2, "bs": BS, "seq": SEQ,
        "sigma": a.sigma, "corpus": corpus_meta, "init_sha_16": init_sha,
        "sched_hash": sched_hash, "sum_lr": sum(lr(t) for t in range(1, T + 1)),
        "wall_sec": time.time() - t0,
        "max_vram_gb": (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else None}
json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
print(f"[TCLOCK] DONE {a.arm} s{a.seed}: {T} steps, {meta['wall_sec']:.0f}s", flush=True)
