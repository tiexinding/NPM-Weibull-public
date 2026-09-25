#!/usr/bin/env python
"""IVN 训练 v3 (260912) — ivn_train_v2.py 的超集. v2 的七个臂 (base/flatv/flatv_rn/nomom/ropeperm/nope/freqscale)
在默认参数下与 v2 逐位一致 (新增代码只在 --arm vmod / --arm mmod / --dump_tensors / --act_scale 下生效).

v3 新增:
  --arm vmod --vmode {rowflat,colflat,fact,perm,permg,scalar} [--rn {none,row,col}] [--vperm_seed S]
        V 干预: 在 opt.step() 后按 flatv 同样的路径撤销标准更新, 用改造过的分母 Ṽ 重做.
        Vh = exp_avg_sq/bc2 (逐坐标 hat V), Mh = exp_avg/bc1; U = Mh/(sqrt(Ṽ)+eps).
          rowflat  Ṽ_ij = mean_j Vh_ij                       (= 现有 flatv)
          colflat  Ṽ_ij = mean_i Vh_ij                       (对称轴)
          fact     Ṽ_ij = r_i c_j / V̄  (r_i=mean_j, c_j=mean_i, V̄=mean)   (Adafactor 式秩1, 精确保行列均值)
          perm     Ṽ_ij = Vh_{i,π_i(j)}   每行一个固定置换 (每行多重集不变)
          permg    Ṽ_ij = Vh_{i,π(j)}     所有行共用一个固定置换
          scalar   Ṽ_ij = mean(Vh)                            (强对照)
        --rn row/col: 把每行/每列的更新范数匹配回标准 Adam 该行/列的范数 (隔离"坐标重加权"与"更新暴露量").
  --arm mmod --mmode {mshort,mreset,sham} --m_window START,END [--rn {none,row}]
        M 干预: mshort = 窗口内用当前 G 代替 Mh 作分子 (V 正常累积); mreset = 窗口起点清零 exp_avg 并对
        后续步使用自 reset 起算的局部 bias correction; sham = 与 mreset 完全相同的保存/日志路径但不清零
        (阴性对照, 与 base 逐位一致).
  --dump_tensors STEPS   在这些步的 optimizer step 之前保存完整张量 {out}/tensors/step{t}.pt:
        每个分析矩阵的 W (step 前), G (.grad), Mh, Vh (该步将要使用的 bias-corrected 值), 以及
        lr/beta1/beta2/eps/wd/step. 供同状态 shadow operator 分析. 单点约 300 MB (fp32).
  --act_scale kind:axis:frac:factor [--act_seed S]
        激活侧干预: forward pre-hook 对残差流输入的固定 frac 比例通道乘 factor.
        只作用于以残差流为输入的投影 (q/k/v/gate/up); o/down 的输入不是残差流, 不作用.
        S7 自检: 关闭 hook 后 logits 与 base 逐位一致.

v2 (260911) 的说明:
  ivn_train.py 的超集. 四个旧臂 (base/flatv/nomom/ropeperm) 在默认参数下与 v1 逐位一致.

新增:
  --arm nope            关闭 RoPE (rotary forward 返回 cos=1, sin=0; 位置无关). preflight S5: rotary 内部拉伸位置后 logits 逐位不变 (RoPE 模型则变).
  --arm freqscale --alpha A   log 频率绕几何平均缩放 log w'_p = log w̄ + A (log w_p - log w̄); A=1 与 base 逐位一致 (不改 buffer).
  --logger [--log_every 200 --log_win 20]   尺度场生成链 side-car (scalefield_logger.py), 只读, 不改轨迹.
  --data_arm {D1,D2,D3,D4} --data_seed S [--extend_base T0] [--held_n N]
                        用 tclock_train.py 的顺序 f-shuffle/tile 数据流替代默认随机有放回采样 (partial_shuffle/tile_to 逐字复制;
                        SeedSequence [9900, data_seed, f*100]; ρ=64 ⇒ 64× 重访). 未指定 data_arm 时采样与 v1 完全相同.
  --resume              从 {out}/resume.pt (model+opt+rng+累加器, 只留最近一份, 在每个 ckpt 步写) 续跑.
  --ckpt_steps 默认 0,200,400,800,1600,3200,5000,10000,15000,20000,25000,30000 (轻量: 分析矩阵 fp32 + energy + mv, 与 v1 同格式).
  --save_final          总是保存 full_model_final.pt (v1 只在 T>=HOR 保存; 测试用).
  --loss_every N        loss.jsonl 记录间隔 (默认 50, 与 v1 同; 测试用 1).
"""
import os, math, json, time, hashlib, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
from scipy.stats import norm
from transformers import LlamaConfig, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["base", "flatv", "nomom", "ropeperm", "nope", "freqscale", "flatv_rn",
                                                "vmod", "mmod"])  # flatv_rn (260912): row-norm-matched flat-v; vmod/mmod: v3
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
ap.add_argument("--ckpt_steps", default="0,200,400,800,1600,3200,5000,10000,15000,20000,25000,30000")
# v2
ap.add_argument("--alpha", type=float, default=1.0, help="freqscale: log-frequency scaling about the geometric mean")
ap.add_argument("--logger", action="store_true")
ap.add_argument("--log_every", type=int, default=200)
ap.add_argument("--log_win", type=int, default=20)
ap.add_argument("--data_arm", default=None, choices=[None, "D1", "D2", "D3", "D4"])
ap.add_argument("--data_seed", type=int, default=None)
ap.add_argument("--extend_base", type=int, default=0, help="data_arm 扩展模式: 语料按 T0 构造 (保 CRN), 之后按 tclock 规则延展")
ap.add_argument("--held_n", type=int, default=524288)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--save_final", action="store_true")
ap.add_argument("--loss_every", type=int, default=50)
ap.add_argument("--full_snapshots", dest="full_snapshots", action="store_true", default=True)
ap.add_argument("--no_full_snapshots", dest="full_snapshots", action="store_false")
# v3
ap.add_argument("--vmode", default=None, choices=["rowflat", "colflat", "fact", "perm", "permg", "scalar"])
ap.add_argument("--rn", default="none", choices=["none", "row", "col"], help="update norm matching axis")
ap.add_argument("--vperm_seed", type=int, default=12345)
ap.add_argument("--mmode", default=None, choices=["mshort", "mreset", "sham"])
ap.add_argument("--m_window", default=None, help="START,END (inclusive)")
ap.add_argument("--dump_tensors", default="", help="comma-separated steps; dump W/G/Mh/Vh before opt.step()")
ap.add_argument("--act_scale", default=None, help="kind:axis:frac:factor, e.g. resid:col:0.1:2.0")
ap.add_argument("--act_seed", type=int, default=777)
a = ap.parse_args()

ARM = a.arm
# ---- v3 argument validation ----
if ARM == "vmod":
    assert a.vmode is not None, "--arm vmod requires --vmode"
if ARM == "mmod":
    assert a.mmode is not None and a.m_window is not None, "--arm mmod requires --mmode and --m_window"
if a.vmode is not None: assert ARM == "vmod", "--vmode only with --arm vmod"
if a.mmode is not None: assert ARM == "mmod", "--mmode only with --arm mmod"
if a.rn != "none": assert ARM in ("vmod", "mmod"), "--rn only with --arm vmod/mmod (flatv_rn is the legacy row-matched arm)"
if ARM == "mmod" and a.rn == "col": raise SystemExit("--rn col is not defined for --arm mmod")
M_WIN = tuple(int(x) for x in a.m_window.split(",")) if a.m_window else None
DUMP_STEPS = sorted(set(int(x) for x in a.dump_tensors.split(",") if x.strip())) if a.dump_tensors.strip() else []
ARM_TAG = ARM if ARM not in ("vmod", "mmod") else (f"vmod:{a.vmode}:rn={a.rn}" if ARM == "vmod" else f"mmod:{a.mmode}:{a.m_window}:rn={a.rn}")
os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/ckpt", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ETA, LWD = a.eta, a.lwd
B1 = 0.0 if ARM == "nomom" else 0.9
B2, EPS = a.beta2, 1e-8
T, HOR, WARM, EMIN, BS, SEQ = a.steps, a.horizon, a.warm, 0.1, a.bs, a.seq
CKPT_STEPS = sorted(set(int(x) for x in a.ckpt_steps.split(",") if int(x) <= T or int(x) == 0))
SCRIPT_SHA = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
              for f in [__file__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scalefield_logger.py")]
              if os.path.exists(f)}

PI = [20,17,5,14,28,27,4,15,2,18,16,1,23,26,7,13,12,29,6,22,9,11,31,19,21,0,8,30,3,25,24,10]

def lr(t):
    if t < WARM: return ETA * t / WARM
    f = (t - WARM) / (HOR - WARM)
    return ETA * EMIN + 0.5 * ETA * (1 - EMIN) * (1 + math.cos(math.pi * min(f, 1.0)))

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

# ---- 模型 ----
torch.manual_seed(a.seed)
cfg = LlamaConfig.from_json_file(a.config); model = LlamaForCausalLM(cfg).to(dev)
def rope_theta_of(c):
    v = getattr(c, "rope_theta", None)
    if v is None and getattr(c, "rope_parameters", None): v = c.rope_parameters.get("rope_theta")
    return float(v) if v is not None else None
ROPE_THETA = rope_theta_of(cfg)

_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def is_analysis(n): return any(f"{k}.weight" in n for k in _KINDS)
ANA = sorted(n for n, _ in model.named_parameters() if is_analysis(n))
pmap = dict(model.named_parameters())
assert len(ANA) == cfg.num_hidden_layers * 7, f"分析矩阵数 {len(ANA)} ≠ {cfg.num_hidden_layers * 7}"

# ================== v3: V 干预 ==================
def _nm_seed(nm):
    """确定性的矩阵名种子 (不用 python hash: 跨进程不稳定)."""
    return int(hashlib.sha256(nm.encode()).hexdigest()[:8], 16)

VPERM = {}          # nm -> LongTensor, perm: (out,in) 每行一个置换; permg: (in,) 全行共用
VPERM_META = {}
if ARM == "vmod" and a.vmode in ("perm", "permg"):
    for nm in ANA:
        out_d, in_d = pmap[nm].shape
        rs = np.random.default_rng(np.random.SeedSequence([a.vperm_seed, _nm_seed(nm)]))
        if a.vmode == "permg":
            pv = rs.permutation(in_d)
            VPERM[nm] = torch.tensor(pv, dtype=torch.long, device=dev)
            VPERM_META[nm] = {"mode": "permg", "n_fixed_points": int((pv == np.arange(in_d)).sum()), "in": int(in_d)}
        else:
            pm_ = np.stack([rs.permutation(in_d) for _ in range(out_d)])
            VPERM[nm] = torch.tensor(pm_, dtype=torch.long, device=dev)
            VPERM_META[nm] = {"mode": "perm", "n_fixed_points_total": int((pm_ == np.arange(in_d)[None, :]).sum()),
                              "shape": [int(out_d), int(in_d)]}

def v_tilde(nm, vhat):
    """vhat: bias-corrected hat-V (float64). 返回改造后的分母 Ṽ (同 shape 或可广播)."""
    m = a.vmode
    if m == "rowflat": return vhat.mean(dim=1, keepdim=True)
    if m == "colflat": return vhat.mean(dim=0, keepdim=True)
    if m == "fact":
        r = vhat.mean(dim=1, keepdim=True); c = vhat.mean(dim=0, keepdim=True)
        return r * c / vhat.mean()
    if m == "perm":  return torch.gather(vhat, 1, VPERM[nm])
    if m == "permg": return vhat[:, VPERM[nm]]
    if m == "scalar": return vhat.mean().expand_as(vhat)
    raise ValueError(m)

def norm_match(u_t, u_adam, mode):
    """把 u_t 的行/列范数匹配回 u_adam 的对应范数 (不改行/列内的相对方向)."""
    if mode == "row":
        s = u_adam.norm(dim=1, keepdim=True) / (u_t.norm(dim=1, keepdim=True) + 1e-30)
    elif mode == "col":
        s = u_adam.norm(dim=0, keepdim=True) / (u_t.norm(dim=0, keepdim=True) + 1e-30)
    else:
        return u_t
    return u_t * s

RN_ERR = {}          # nm -> {"med": , "max": } 范数匹配相对误差 (数值对账, 构造上应 ~0)
def rn_err(u_t, u_adam, mode):
    if mode == "row":
        na, nt = u_adam.norm(dim=1), u_t.norm(dim=1)
    elif mode == "col":
        na, nt = u_adam.norm(dim=0), u_t.norm(dim=0)
    else:
        return None
    e = ((nt - na).abs() / (na + 1e-30))
    return {"med": float(e.median()), "max": float(e.max())}

# ================== v3: 激活侧干预 ==================
ACT_ON = [True]           # 可切换, 供 S7 自检
ACT_META = None
_ACT_HANDLES = []
if a.act_scale is not None:
    _p = a.act_scale.split(":")
    assert len(_p) == 4, "--act_scale 格式: kind:axis:frac:factor"
    ak, ax, afrac, afac = _p[0], _p[1], float(_p[2]), float(_p[3])
    assert ak == "resid" and ax == "col", "当前只实现 resid:col (残差流输入通道)"
    d_model = int(cfg.hidden_size)
    _rs = np.random.default_rng(np.random.SeedSequence([a.act_seed, 4242]))
    n_sel = max(1, int(round(afrac * d_model)))
    sel = np.sort(_rs.choice(d_model, size=n_sel, replace=False))
    scale_vec = torch.ones(d_model, dtype=torch.float32, device=dev)
    scale_vec[torch.tensor(sel, dtype=torch.long, device=dev)] = afac
    _ACT_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")   # 输入是残差流的投影
    def _mk_hook(vec):
        def hook(mod, args):
            if not ACT_ON[0]: return None
            x = args[0]
            return (x * vec.to(x.dtype),) + tuple(args[1:])
        return hook
    _n_hooked = 0
    for mod_name, mod in model.named_modules():
        if any(mod_name.endswith(k) for k in _ACT_TARGETS) and isinstance(mod, torch.nn.Linear):
            _ACT_HANDLES.append(mod.register_forward_pre_hook(_mk_hook(scale_vec)))
            _n_hooked += 1
    ACT_META = {"spec": a.act_scale, "act_seed": a.act_seed, "d_model": d_model, "n_selected": int(n_sel),
                "factor": afac, "frac": afrac, "selected_channels": [int(i) for i in sel],
                "targets": list(_ACT_TARGETS), "n_modules_hooked": _n_hooked,
                "note": "缩放残差流输入通道 = 等价缩放该列的有效权重; 该臂的权重列场读数与 base 不同尺度, 可比性受限"}
    assert _n_hooked == cfg.num_hidden_layers * 5, f"act hook 数 {_n_hooked} ≠ {cfg.num_hidden_layers * 5}"
    ACT_ON[0] = False        # preflight 的 out0 是 base (无 hook); S7 之后再打开

def apply_rope_perm(mdl, perm):
    perm_t = torch.tensor(perm, dtype=torch.long, device=dev) if not isinstance(perm, torch.Tensor) else perm
    rope = mdl.model.rotary_emb
    assert hasattr(rope, "inv_freq"), "rotary_emb missing inv_freq"
    orig = rope.inv_freq.clone()
    rope.inv_freq.copy_(orig[perm_t])
    if hasattr(rope, "original_inv_freq"):
        rope.original_inv_freq = rope.inv_freq.clone()
    return 1

# ---- v2: nope / freqscale ----
def apply_nope(mdl):
    """rotary forward -> (cos=1, sin=0): q_embed = q, k_embed = k. 位置无关, 精确."""
    rope = mdl.model.rotary_emb
    orig_fwd = rope.forward
    def fwd(x, position_ids):
        cos, sin = orig_fwd(x, position_ids)
        return torch.ones_like(cos), torch.zeros_like(sin)
    rope.forward = fwd
    return rope

def apply_freqscale(mdl, alpha):
    rope = mdl.model.rotary_emb
    before = rope.inv_freq.detach().double().cpu().numpy().copy()
    if alpha == 1.0:
        return before, before.copy(), {"alpha": 1.0, "modified": False}
    lf = np.log(before); lm = lf.mean()
    new = np.exp(lm + alpha * (lf - lm))
    assert new.shape == before.shape
    assert abs(np.log(new).mean() - lm) < 1e-9, "geometric mean must be preserved"
    rope.inv_freq.copy_(torch.tensor(new, dtype=rope.inv_freq.dtype, device=rope.inv_freq.device))
    if hasattr(rope, "original_inv_freq"):
        rope.original_inv_freq = rope.inv_freq.clone()
    return before, new, {"alpha": alpha, "modified": True,
                         "log_range_before": [float(lf.min()), float(lf.max())],
                         "log_range_after": [float(np.log(new).min()), float(np.log(new).max())]}

# ---- 三族耦合 init (与 v1 逐位同) ----
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

# v2: tclock 顺序数据流 (partial_shuffle / tile_to 逐字复制自 tclock_train.py)
ARMS_DATA = {"D1": (0.00, 1), "D2": (1.00, 1), "D3": (1.00, 64), "D4": (0.00, 64)}
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

corpus_meta = None; held = None; stream = None
if a.data_arm is not None:
    assert a.data_seed is not None, "--data_arm requires --data_seed"
    F_SH, RHO = ARMS_DATA[a.data_arm]
    full = toks
    T0 = a.extend_base if a.extend_base else T
    T0_TOK = T0 * BS * SEQ; T_TOK = T * BS * SEQ
    U = T0_TOK // RHO
    EXT_TOK = T_TOK - T0_TOK
    NEED_EXT = max(EXT_TOK, 0) if RHO == 1 else 0   # 260911: ρ>1 延展为平铺 D 块 (tile_continue), 不消耗母序列
    assert len(full) >= T0_TOK + a.held_n + NEED_EXT + SEQ, \
        f"母序列不足: {len(full)} < {T0_TOK + a.held_n + NEED_EXT + SEQ}"
    rng_sh = np.random.default_rng(np.random.SeedSequence([9900, a.data_seed, int(F_SH * 100)]))
    blk = partial_shuffle(np.asarray(full[:U]).astype(np.int64).copy(), F_SH, rng_sh)
    stream = tile_to(blk, T0_TOK)
    ext_mode = None
    if EXT_TOK > 0:
        if RHO == 1:
            src = np.asarray(full[T0_TOK + a.held_n: T0_TOK + a.held_n + EXT_TOK]).astype(np.int64).copy()
            rng_e = np.random.default_rng(np.random.SeedSequence([9922, a.data_seed, int(F_SH * 100)]))
            ext = partial_shuffle(src, F_SH, rng_e); ext_mode = "fresh_after_held"
        else:
            ext = tile_to(blk, T0_TOK + EXT_TOK)[T0_TOK:]; ext_mode = "tile_continue"
        stream = np.concatenate([stream, ext])
    tail = np.asarray(full[T0_TOK: T0_TOK + a.held_n]).astype(np.int64).copy()
    rng_h = np.random.default_rng(np.random.SeedSequence([9911, a.data_seed, int(F_SH * 100)]))
    held = partial_shuffle(tail, F_SH, rng_h)
    def block_hashes(x, k=1024, n=2000, rng=None):
        if len(x) <= k + 1:
            return {hashlib.sha256(x.tobytes()).hexdigest()}
        idx = (rng or np.random.default_rng(0)).integers(0, len(x) - k, n)
        return {hashlib.sha256(x[i:i + k].tobytes()).hexdigest() for i in idx}
    leak = len(block_hashes(blk, rng=np.random.default_rng(1)) & block_hashes(held, rng=np.random.default_rng(2)))
    corpus_meta = {"data_arm": a.data_arm, "data_seed": a.data_seed, "f_shuffle": F_SH, "rho": RHO,
                   "U": int(U), "T0": int(T0), "T_tok": int(T_TOK), "ext_tok": int(max(EXT_TOK, 0)), "ext_mode": ext_mode,
                   "revisit_x": float(T_TOK / U),
                   "stream_sha256_16": hashlib.sha256(stream.tobytes()).hexdigest()[:16],
                   "held_sha256_16": hashlib.sha256(held.tobytes()).hexdigest()[:16],
                   "leak_blocks_1024": int(leak)}
    if leak > 0:
        open(f"{a.out}/ABORT_heldout_leak", "w").write(str(leak)); raise SystemExit(6)
    def get_batch(log_hash=False):
        raise RuntimeError("use batch(t) for data_arm streams")
    def batch(t, log_hash=False):
        o = (t - 1) * BS * SEQ
        x = stream[o:o + BS * SEQ].reshape(BS, SEQ)
        if log_hash: batch_hashes.append(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:12])
        return torch.tensor(x, dtype=torch.long, device=dev)
else:
    def get_batch(log_hash=False):
        idx = rng.integers(0, len(toks) - SEQ - 1, BS)
        if log_hash: batch_hashes.append(hashlib.sha256(idx.tobytes()).hexdigest()[:12])
        return torch.tensor(np.stack([toks[i:i + SEQ] for i in idx]), dtype=torch.long, device=dev)
    def batch(t, log_hash=False):
        return get_batch(log_hash=log_hash)

@torch.no_grad()
def held_eval():
    if held is None: return None
    model.eval(); n = (len(held) - 1) // (BS * SEQ); tot = 0.0
    for b in range(n):
        x = torch.tensor(held[b * BS * SEQ:(b + 1) * BS * SEQ].reshape(BS, SEQ), dtype=torch.long, device=dev)
        tot += float(model(input_ids=x, labels=x).loss)
    model.train(); return tot / max(n, 1)

# ---- Preflight ----
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

s3_report = None; s5_report = None; s6_report = None
if ARM == "ropeperm":
    loss_identity = step0_loss
    logits_identity = out0.logits.detach()
    n_applied = apply_rope_perm(model, PI)
    out_perm = model(input_ids=px, labels=px)
    loss_perm = float(out_perm.loss)
    model.zero_grad(set_to_none=True)
    s3_logit_maxdiff = float((logits_identity - out_perm.logits.detach()).abs().max())
    s3_loss_diff = abs(loss_perm - loss_identity)
    s3b_pass = s3_logit_maxdiff > 1e-3
    del logits_identity
    inv_freq_after = model.model.rotary_emb.inv_freq.cpu().tolist()
    s3_report = {"loss_identity": loss_identity, "loss_perm": loss_perm, "loss_diff": s3_loss_diff,
                 "logit_maxdiff": s3_logit_maxdiff, "s3b_pass": s3b_pass,
                 "layers_applied": n_applied, "inv_freq_after_layer0": inv_freq_after}
    print(f"[IVN-ropeperm] S3: identity_loss={loss_identity:.6f} perm_loss={loss_perm:.6f} "
          f"loss_diff={s3_loss_diff:.6f} logit_maxdiff={s3_logit_maxdiff:.4f} "
          f"S3b={'PASS' if s3b_pass else 'FAIL'}", flush=True)
    if not s3b_pass:
        open(f"{a.out}/ABORT_S3_perm_no_effect", "w").close(); raise SystemExit(6)
    del out_perm, out0
    if dev == "cuda": torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"[IVN-ropeperm] S3 cleanup: VRAM freed", flush=True)

elif ARM == "nope":
    # S5: positions are probed INSIDE the rotary module (position_ids*2), never through the model's position_ids
    # argument, because create_causal_mask() treats any non-arange position_ids as packed-sequence boundaries.
    # (pre)  RoPE model: stretched rotary positions change the logits (> 1e-3)
    # (post) NoPE model: rotary returns constants, so stretched positions leave the logits bitwise unchanged,
    #        and NoPE logits differ from RoPE logits (> 1e-3) -> the patch is effective.
    with torch.no_grad():
        rope = model.model.rotary_emb
        logits_rope = out0.logits.detach()
        orig_fwd = rope.forward
        rope.forward = (lambda x_, position_ids: orig_fwd(x_, position_ids * 2))
        logits_rope_stretch = model(input_ids=px).logits.detach()
        rope.forward = orig_fwd
        pre_diff = float((logits_rope - logits_rope_stretch).abs().max())
        apply_nope(model)                     # rope.forward -> constants (wraps orig_fwd)
        nope_fwd = rope.forward
        logits_nope = model(input_ids=px).logits.detach()
        rope.forward = (lambda x_, position_ids: nope_fwd(x_, position_ids * 2))
        logits_nope_stretch = model(input_ids=px).logits.detach()
        rope.forward = nope_fwd
        post_diff = float((logits_nope - logits_nope_stretch).abs().max())
        patch_diff = float((logits_nope - logits_rope).abs().max())
        s5_pass = (pre_diff > 1e-3) and (post_diff == 0.0) and (patch_diff > 1e-3)
    s5_report = {"logit_maxdiff_rope_stretched_positions": pre_diff,
                 "logit_maxdiff_nope_stretched_positions": post_diff,
                 "logit_maxdiff_nope_vs_rope": patch_diff, "s5_pass": bool(s5_pass)}
    print(f"[IVN-nope] S5: rope stretch diff={pre_diff:.4f} nope stretch diff={post_diff:.3e} "
          f"nope-vs-rope={patch_diff:.4f} S5={'PASS' if s5_pass else 'FAIL'}", flush=True)
    del logits_rope, logits_rope_stretch, logits_nope, logits_nope_stretch, out0
    if not s5_pass:
        open(f"{a.out}/ABORT_S5_nope", "w").close(); raise SystemExit(7)
    if dev == "cuda": torch.cuda.empty_cache()
    import gc; gc.collect()

elif ARM == "freqscale":
    logits_base = out0.logits.detach()
    inv_before, inv_after, fs_meta = apply_freqscale(model, a.alpha)
    with torch.no_grad():
        logits_fs = model(input_ids=px).logits.detach()
    diff = float((logits_base - logits_fs).abs().max())
    s6_pass = (diff == 0.0) if a.alpha == 1.0 else (diff > 1e-3)
    s6_report = {**fs_meta, "inv_freq_before": inv_before.tolist(), "inv_freq_after": inv_after.tolist(),
                 "logit_maxdiff_vs_base": diff, "s6_pass": bool(s6_pass)}
    print(f"[IVN-freqscale] S6: alpha={a.alpha} logit_maxdiff={diff:.4e} S6={'PASS' if s6_pass else 'FAIL'}", flush=True)
    del logits_base, logits_fs, out0
    if not s6_pass:
        open(f"{a.out}/ABORT_S6_freqscale", "w").close(); raise SystemExit(8)
    if dev == "cuda": torch.cuda.empty_cache()
    import gc; gc.collect()

# ---- v3 S7: act_scale hook 自检 (out0 是 hook 关闭时的 base) ----
s7_report = None
if a.act_scale is not None:
    with torch.no_grad():
        logits_base = model(input_ids=px).logits.detach()
        _o0 = globals().get("out0")
        base_off_diff = float((logits_base - _o0.logits.detach()).abs().max()) if _o0 is not None else 0.0
        ACT_ON[0] = True
        logits_hooked = model(input_ids=px).logits.detach()
        on_diff = float((logits_hooked - logits_base).abs().max())
        ACT_ON[0] = False
        logits_off2 = model(input_ids=px).logits.detach()
        off_diff = float((logits_off2 - logits_base).abs().max())
    s7_pass = (base_off_diff == 0.0) and (on_diff > 1e-3) and (off_diff == 0.0)
    s7_report = {**ACT_META, "logit_maxdiff_hook_on_vs_base": on_diff,
                 "logit_maxdiff_hook_off_vs_base": off_diff,
                 "logit_maxdiff_out0_vs_base": base_off_diff, "s7_pass": bool(s7_pass)}
    print(f"[IVN-{ARM_TAG}] S7 act_scale: on-vs-base={on_diff:.4f} off-vs-base={off_diff:.3e} "
          f"S7={'PASS' if s7_pass else 'FAIL'}", flush=True)
    del logits_base, logits_hooked, logits_off2
    if not s7_pass:
        open(f"{a.out}/ABORT_S7_act_scale", "w").close(); raise SystemExit(9)
    ACT_ON[0] = True          # 训练期打开
    if dev == "cuda": torch.cuda.empty_cache()

zg_ok, zg_dev = zero_grad_decay_check(lr(max(WARM, 1)))

preflight = {"arm": ARM, "family": a.family, "seed": a.seed, "beta1": B1,
             "config": os.path.basename(a.config), "rope_theta": ROPE_THETA,
             "step0_loss": step0_loss,
             "hidden_rms_per_layer": hs_rms,
             "grad_rms_median": float(np.median(list(grad_rms.values()))),
             "sign_agree_median": float(np.median([v["sign_agree_vs_gaussian"] for v in init_report.values()])),
             "zero_grad_decay_check": {"pass": bool(zg_ok), "max_dev": zg_dev},
             "s3_ropeperm": s3_report, "s5_nope": s5_report, "s6_freqscale": s6_report,
             "s7_act_scale": s7_report,
             "tokens": tok_meta, "corpus": corpus_meta,
             "pi": PI if ARM == "ropeperm" else None,
             "alpha": a.alpha if ARM == "freqscale" else None,
             "logger": {"enabled": a.logger, "every": a.log_every, "win": a.log_win},
             "v3": {"arm_tag": ARM_TAG, "vmode": a.vmode, "rn": a.rn, "vperm_seed": a.vperm_seed,
                    "vperm_meta": VPERM_META, "mmode": a.mmode, "m_window": M_WIN,
                    "dump_steps": DUMP_STEPS, "act_scale": ACT_META},
             "ckpt_steps": CKPT_STEPS, "script_sha256_16": SCRIPT_SHA}
json.dump({"preflight": preflight, "init_per_matrix": init_report},
          open(f"{a.out}/preflight.json", "w"), indent=1)
print(f"[IVN-{ARM}] preflight: loss={step0_loss:.4f} "
      f"sign_agree_med={preflight['sign_agree_median']:.4f} revisit={tok_meta['revisit_x']}x", flush=True)
if not zg_ok:
    open(f"{a.out}/ABORT_S2_zero_grad_decay", "w").close(); raise SystemExit(4)

# ---- Optimizer ----
opt = torch.optim.AdamW(model.parameters(), lr=ETA, weight_decay=LWD, betas=(B1, B2), eps=EPS)

acc = {nm: {q: {"row": torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev),
               "col": torch.zeros(pmap[nm].shape[1], dtype=torch.float64, device=dev)}
            for q in ("tot", "adp", "dec", "cross")} for nm in ANA}
mv = {nm: {q: torch.zeros(pmap[nm].shape[0], dtype=torch.float64, device=dev)
           for q in ("Ag", "Au", "SF", "prev_sign")} for nm in ANA}

# ---- v2: logger ----
logger = None
if a.logger:
    from scalefield_logger import ScaleFieldLogger
    logger = ScaleFieldLogger(model, ANA, pmap, opt, a.out, every=a.log_every, win=a.log_win,
                              beta1=B1, beta2=B2, eps=EPS, lwd=LWD, total_steps=T)

def save_ckpt(t):
    torch.save({nm: pmap[nm].detach().to(torch.float32).cpu() for nm in ANA}, f"{a.out}/ckpt/step{t}.pt")
    dump = {}
    for nm in ANA:
        for q in ("tot", "adp", "dec", "cross"):
            for ax in ("row", "col"):
                dump[f"{nm}|{q}|{ax}"] = acc[nm][q][ax].cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/energy_step{t}.npz", **dump)
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
            if ARM in ("flatv", "flatv_rn", "vmod") and t > 0:
                sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                bc1c = 1 - B1 ** sv; bc2c = 1 - B2 ** sv
                mh = m_.double() / bc1c; vh = v_.double() / bc2c
                u_adam = mh / (vh.sqrt() + EPS)
                vbar = v_tilde(nm, vh) if ARM == "vmod" else vh.mean(dim=1, keepdim=True)
                u_flat = mh / (vbar.sqrt() + EPS)
                r = u_flat.norm(dim=1) / (u_adam.norm(dim=1) + 1e-30)
                mvd[f"{nm}|exposure_ratio"] = r.cpu().numpy()
    np.savez_compressed(f"{a.out}/ckpt/mv_step{t}.npz", **mvd)
    if a.full_snapshots:
        os.makedirs(f"{a.out}/full", exist_ok=True)
        sd16 = {k: (v.detach().to(torch.float16).cpu() if v.is_floating_point() else v.detach().cpu())
                for k, v in model.state_dict().items()}
        fp = f"{a.out}/full/step{t}.pt"; torch.save(sd16, fp)
        rope = model.model.rotary_emb
        side = {"step": t, "arm": ARM, "dtype": "float16", "config": os.path.basename(a.config), "rope_theta": ROPE_THETA,
                "inv_freq": rope.inv_freq.detach().float().cpu().tolist(),
                "pi": PI if ARM == "ropeperm" else None, "alpha": a.alpha if ARM == "freqscale" else None,
                "nope": ARM == "nope", "sha256_16": hashlib.sha256(open(fp, "rb").read()).hexdigest()[:16]}
        json.dump(side, open(f"{a.out}/full/step{t}.json", "w"))
        full_snap_log.append({"step": t, "sha256_16": side["sha256_16"]})

DUMP_LOG = []
def dump_tensors(t, cur):
    """在 opt.step() 之前调用: 保存 W(step 前), G(.grad), 以及该步将要使用的 bias-corrected Mh/Vh.
    state 里此刻是 M_{t-1}, V_{t-1} (step 计数 t-1), 所以
        M_t = b1*M_{t-1} + (1-b1)*G_t,  Mh = M_t/(1-b1^t)
        V_t = b2*V_{t-1} + (1-b2)*G_t^2, Vh = V_t/(1-b2^t)
    与 torch AdamW 的实现一致. M_{t-1}/V_{t-1} 可由 (Mh,Vh,G) 反解, 不另存."""
    os.makedirs(f"{a.out}/tensors", exist_ok=True)
    bc1 = 1 - B1 ** t; bc2 = 1 - B2 ** t
    blob = {}
    for nm in ANA:
        p = pmap[nm]; st = opt.state.get(p, {})
        g = p.grad.detach().float()
        m_prev = st.get("exp_avg"); v_prev = st.get("exp_avg_sq")
        m_prev = torch.zeros_like(g) if m_prev is None else m_prev.detach().float()
        v_prev = torch.zeros_like(g) if v_prev is None else v_prev.detach().float()
        m_t = B1 * m_prev + (1 - B1) * g
        v_t = B2 * v_prev + (1 - B2) * g * g
        blob[f"{nm}|W"] = p.detach().float().cpu()
        blob[f"{nm}|G"] = g.cpu()
        blob[f"{nm}|Mh"] = (m_t / bc1).cpu()
        blob[f"{nm}|Vh"] = (v_t / bc2).cpu()
    blob["__meta__"] = {"step": t, "lr": cur, "beta1": B1, "beta2": B2, "eps": EPS, "wd": LWD,
                        "bc1": bc1, "bc2": bc2, "arm": ARM_TAG, "dtype": "float32",
                        "state_step_before": int(opt.state.get(pmap[ANA[0]], {}).get("step", 0) or 0),
                        "note": "W is pre-step; Mh/Vh are the bias-corrected values this step's update uses"}
    fp = f"{a.out}/tensors/step{t}.pt"; torch.save(blob, fp)
    sz = os.path.getsize(fp)
    DUMP_LOG.append({"step": t, "bytes": sz, "mb": round(sz / 2**20, 1),
                     "sha256_16": hashlib.sha256(open(fp, "rb").read()).hexdigest()[:16]})
    print(f"[IVN-{ARM_TAG}] dump_tensors step{t}: {sz/2**20:.0f} MB", flush=True)

M_RESET_LOG = []
def save_resume(t, loss200):
    tmp = f"{a.out}/.resume.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state() if dev == "cuda" else None),
                "np_rng": rng.bit_generator.state,
                "acc": {nm: {q: {ax: acc[nm][q][ax].cpu() for ax in ("row", "col")} for q in acc[nm]} for nm in ANA},
                "mv": {nm: {q: mv[nm][q].cpu() for q in mv[nm]} for nm in ANA},
                "batch_hashes": list(batch_hashes), "loss200": loss200, "step": t,
                "held_log": held_log, "full_snap_log": full_snap_log}, tmp)
    os.replace(tmp, f"{a.out}/resume.pt")

held_log = []; full_snap_log = []
START = 1; loss200 = None
if a.resume:
    rp = f"{a.out}/resume.pt"
    assert os.path.exists(rp), "--resume 但无 resume.pt"
    sd = torch.load(rp, map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
    torch.set_rng_state(sd["torch_rng"].cpu())
    if dev == "cuda" and sd["cuda_rng"] is not None: torch.cuda.set_rng_state(sd["cuda_rng"].cpu())
    rng.bit_generator.state = sd["np_rng"]
    for nm in ANA:
        for q in acc[nm]:
            for ax in ("row", "col"): acc[nm][q][ax] = sd["acc"][nm][q][ax].to(dev)
        for q in mv[nm]: mv[nm][q] = sd["mv"][nm][q].to(dev)
    batch_hashes[:] = sd["batch_hashes"]; loss200 = sd["loss200"]; held_log = sd.get("held_log", []); full_snap_log = sd.get("full_snap_log", [])
    START = sd["step"] + 1
    if logger is not None: logger.opt = opt
    print(f"[IVN-{ARM}] resume from step{sd['step']}", flush=True)
else:
    save_ckpt(0)
    if held is not None: held_log.append({"step": 0, "held_loss": held_eval()})
    save_resume(0, loss200)

# ---- 训练主循环 (v1 结构逐位保留; 新增: logger 三个入口, batch(t), resume, held) ----
t0 = time.time(); s1_checked = False; id_checked = False; t_win = 0.0; n_win = 0; t_plain = 0.0; n_plain = 0
for t in range(START, T + 1):
    ts = time.time()
    cur = lr(t)
    for g in opt.param_groups: g["lr"] = cur
    W_pre = {nm: pmap[nm].detach().clone() for nm in ANA}

    opt.zero_grad(set_to_none=True)
    if logger is not None: logger.before_forward(t)
    x = batch(t, log_hash=(t <= 5 or (a.data_arm is not None and t <= 20)))
    out = model(input_ids=x, labels=x); loss = out.loss
    loss.backward()
    if logger is not None: logger.after_backward(t)

    with torch.no_grad():
        for nm in ANA:
            rg = (pmap[nm].grad.double() * W_pre[nm].double()).sum(1)
            mv[nm]["Ag"] += cur * rg
            sg = torch.sign(rg)
            mv[nm]["SF"] += ((sg != mv[nm]["prev_sign"]) & (mv[nm]["prev_sign"] != 0)).double()
            mv[nm]["prev_sign"] = sg

    # ---- v3: dump tensors (opt.step() 之前) ----
    if t in DUMP_STEPS: dump_tensors(t, cur)

    # ---- v3: mreset 在窗口起点清零 exp_avg (opt.step() 之前) ----
    if ARM == "mmod" and a.mmode == "mreset" and t == M_WIN[0]:
        with torch.no_grad():
            n_z = 0
            for nm in ANA:
                st = opt.state.get(pmap[nm], {})
                if st.get("exp_avg") is not None:
                    st["exp_avg"].zero_(); n_z += 1
        M_RESET_LOG.append({"step": t, "n_zeroed": n_z})
        print(f"[IVN-{ARM_TAG}] mreset: exp_avg zeroed at step{t} ({n_z} matrices)", flush=True)
    elif ARM == "mmod" and a.mmode == "sham" and t == M_WIN[0]:
        # 阴性对照: 走同样的分支与日志, 但不改任何状态 (与 base 逐位一致)
        with torch.no_grad():
            n_z = sum(1 for nm in ANA if opt.state.get(pmap[nm], {}).get("exp_avg") is not None)
        M_RESET_LOG.append({"step": t, "n_zeroed": 0, "sham": True, "n_would_zero": n_z})
        print(f"[IVN-{ARM_TAG}] sham: no-op at step{t} ({n_z} matrices would have been zeroed)", flush=True)

    opt.step()

    # ---- v3: V / M 干预 (与 flatv 同路径: 撤销标准更新, 用改造过的算子重做) ----
    U_APPLIED = {}
    if ARM in ("vmod", "mmod"):
        # sham 永不重做 (必须与 base 逐位一致)
        in_win = (ARM == "mmod") and (a.mmode != "sham") and (M_WIN[0] <= t <= M_WIN[1])
        # mreset 的局部 bias correction: reset 后用自 reset 起算的步数, 直到与全局 bc1 数值上无区别
        mreset_local = (ARM == "mmod" and a.mmode == "mreset" and t >= M_WIN[0]
                        and abs(B1 ** (t - M_WIN[0] + 1) - B1 ** t) > 1e-13)
        need_redo = (ARM == "vmod") or in_win or mreset_local
        if need_redo:
            with torch.no_grad():
                for nm in ANA:
                    st = opt.state[pmap[nm]]
                    sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                    bc1 = 1 - B1 ** sv; bc2 = 1 - B2 ** sv
                    vhat = st["exp_avg_sq"].double() / bc2
                    mhat_std = st["exp_avg"].double() / bc1
                    u_adam = mhat_std / (vhat.sqrt() + EPS)
                    if ARM == "vmod":
                        u_t = mhat_std / (v_tilde(nm, vhat).sqrt() + EPS)
                    else:                                   # mmod
                        if a.mmode == "mshort" and in_win:
                            u_t = pmap[nm].grad.detach().double() / (vhat.sqrt() + EPS)
                        elif a.mmode == "mreset":
                            bc1_loc = 1 - B1 ** (t - M_WIN[0] + 1)
                            u_t = (st["exp_avg"].double() / bc1_loc) / (vhat.sqrt() + EPS)
                        else:                               # sham 不该进这里
                            u_t = u_adam
                    if a.rn != "none":
                        u_t = norm_match(u_t, u_adam, a.rn)
                        if t == 250: RN_ERR[nm] = rn_err(u_t, u_adam, a.rn)
                    pmap[nm].copy_((W_pre[nm].double() * (1 - cur * LWD) - cur * u_t).float())
                    if t == 250: U_APPLIED[nm] = u_t

    if ARM in ("flatv", "flatv_rn"):
        with torch.no_grad():
            for nm in ANA:
                st = opt.state[pmap[nm]]
                sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                bc1 = 1 - B1 ** sv; bc2 = 1 - B2 ** sv
                mhat = st["exp_avg"].double() / bc1
                vhat = st["exp_avg_sq"].double() / bc2
                vbar = vhat.mean(dim=1, keepdim=True)
                u_flat = mhat / (vbar.sqrt() + EPS)
                if ARM == "flatv_rn":
                    # row-norm-matched flat-v (260912): keep flat-v's within-row direction, rescale each row of the
                    # update to the row norm the standard Adam direction would have had (isolates coordinate re-weighting
                    # from the change in per-row update exposure)
                    u_adam = mhat / (vhat.sqrt() + EPS)
                    scale = u_adam.norm(dim=1, keepdim=True) / (u_flat.norm(dim=1, keepdim=True) + 1e-30)
                    u_flat = u_flat * scale
                pmap[nm].copy_((W_pre[nm].double() * (1 - cur * LWD) - cur * u_flat).float())

    if logger is not None: logger.after_step(t, cur, W_pre)

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
            if t == 250 and not s1_checked:
                st = opt.state.get(pmap[nm], {})
                if st.get("exp_avg") is not None:
                    sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                    bc1c = 1 - B1 ** sv; bc2c = 1 - B2 ** sv
                    mh = st["exp_avg"].double() / bc1c
                    vh = st["exp_avg_sq"].double() / bc2c
                    if nm in U_APPLIED:                      # v3: vmod/mmod 用实际施加的 u
                        u_check = U_APPLIED[nm]
                    elif ARM in ("flatv", "flatv_rn"):
                        vb = vh.mean(dim=1, keepdim=True); u_check = mh / (vb.sqrt() + EPS)
                        if ARM == "flatv_rn":
                            u_ad = mh / (vh.sqrt() + EPS)
                            u_check = u_check * (u_ad.norm(dim=1, keepdim=True) / (u_check.norm(dim=1, keepdim=True) + 1e-30))
                    else:
                        u_check = mh / (vh.sqrt() + EPS)
                    au_state = cur * (u_check * W_pre[nm].double()).sum(1)
                    rel = float((au_state - au_inc).abs().max() / (au_inc.abs().max() + 1e-30))
                    S4_TOL = float(os.environ.get("S4_TOL", "5e-4"))   # 260911: 1e-4 → 5e-4 (4090 fp32: down_proj rel 1.28e-4 vs A800 4.9e-5); value recorded
                    s4_rel_log = globals().setdefault("S4_REL_LOG", {}); s4_rel_log[nm] = rel
                    if rel > S4_TOL:
                        tag = f"S1_{ARM}" if ARM in ("flatv", "flatv_rn", "nomom", "vmod", "mmod") else "S4"
                        open(f"{a.out}/ABORT_{tag}_identity", "w").write(f"{nm} rel={rel:.2e} tol={S4_TOL:.1e}\n")
                        raise SystemExit(5)
                    print(f"[IVN-{ARM}] S1/S4 PASS ({nm}: rel={rel:.2e}, tol={S4_TOL:.1e})", flush=True)
    if t == 250: s1_checked = True

    if a.dryrun and t == 3 and not id_checked:
        nm0 = ANA[0]; A0 = acc[nm0]
        lhs = A0["tot"]["row"]; rhs = A0["adp"]["row"] + A0["dec"]["row"] + 2 * A0["cross"]["row"]
        ok = torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-18)
        print(f"[IVN-{ARM}] 分解恒等式 @step3: {'PASS' if ok else 'FAIL'} (注: 残差定义恒真, 仅校累加精度)", flush=True)
        id_checked = True

    lv = float(loss)
    if not math.isfinite(lv):
        open(f"{a.out}/ABORT_nonfinite_step{t}", "w").close(); raise SystemExit(3)
    if t == 200: loss200 = lv
    if loss200 and t > 200 and lv > 10 * loss200:
        open(f"{a.out}/ABORT_catastrophic_step{t}", "w").close(); raise SystemExit(3)

    dt = time.time() - ts
    if logger is not None and logger.in_window(t): t_win += dt; n_win += 1
    else: t_plain += dt; n_plain += 1

    if t % a.loss_every == 0:
        with open(f"{a.out}/loss.jsonl", "a") as f:
            f.write(json.dumps({"step": t, "loss": lv, "eta": cur,
                                "upd_w_ratio_med": float(np.median(ratios)) if ratios else None}) + "\n")
        if t % 50 == 0:
            print(f"[IVN-{ARM}] step{t} loss={lv:.4f} "
                  f"u/w={np.median(ratios) if ratios else float('nan'):.3e} ({(time.time() - t0) / max(t - START + 1, 1):.3f}s/step)", flush=True)

    if t in CKPT_STEPS:
        save_ckpt(t)
        if held is not None:
            if logger is not None: logger.active = False
            held_log.append({"step": t, "held_loss": held_eval()})
        save_resume(t, loss200)

if logger is not None: logger.close()
if T >= HOR or a.save_final:
    torch.save(model.state_dict(), f"{a.out}/full_model_final.pt")

sum_lr = sum(lr(t) for t in range(1, T + 1))
meta = {"s4_rel_by_matrix": globals().get("S4_REL_LOG", {}), "s4_tol": float(os.environ.get("S4_TOL", "5e-4")), "arm": ARM, "family": a.family, "seed": a.seed,
        "config": os.path.basename(a.config), "rope_theta": ROPE_THETA,
        "hidden_size": int(cfg.hidden_size), "num_hidden_layers": int(cfg.num_hidden_layers),
        "steps": T, "horizon": HOR, "warm": WARM, "emin": EMIN,
        "beta1": B1, "beta2": B2, "eta_peak": ETA, "lwd": LWD,
        "bs": BS, "seq": SEQ, "sigma": a.sigma,
        "sum_lr": sum_lr, "cum_decay_exposure": LWD * sum_lr,
        "sum_log1m_eta_lambda": sum(math.log1p(-lr(t) * LWD) for t in range(1, T + 1)),
        "zero_grad_decay_check": {"pass": bool(zg_ok), "max_dev": zg_dev},
        "s3_ropeperm": s3_report, "s5_nope": s5_report, "s6_freqscale": s6_report, "s7_act_scale": s7_report,
        "pi": PI if ARM == "ropeperm" else None,
        "alpha": a.alpha if ARM == "freqscale" else None,
        "v3": {"arm_tag": ARM_TAG, "vmode": a.vmode, "rn": a.rn, "vperm_seed": a.vperm_seed,
               "vperm_meta": VPERM_META, "mmode": a.mmode, "m_window": M_WIN,
               "m_reset_log": M_RESET_LOG,
               "rn_match_rel_err_by_kind": ({k: {"med": float(np.median([v["med"] for n, v in RN_ERR.items() if k in n])),
                                                 "max": float(max(v["max"] for n, v in RN_ERR.items() if k in n))}
                                             for k in _KINDS if any(k in n for n in RN_ERR)} if RN_ERR else None),
               "dump_tensors": DUMP_LOG, "act_scale": ACT_META},
        "corpus": corpus_meta, "held_log": held_log,
        "full_snapshots": {"enabled": a.full_snapshots, "dtype": "float16", "dir": "full/", "steps": full_snap_log},
        "batch_hashes_first5": batch_hashes[:5], "batch_hashes_first20": batch_hashes[:20],
        "logger": (logger.meta() if logger is not None else {"enabled": False}),
        "step_time_plain_s": (t_plain / n_plain) if n_plain else None,
        "step_time_logger_window_s": (t_win / n_win) if n_win else None,
        "ckpt_steps": CKPT_STEPS, "resumed_from": (START - 1) if a.resume else None,
        "script_sha256_16": SCRIPT_SHA,
        "wall_sec": time.time() - t0,
        "max_vram_gb": (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else None}
json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
print(f"[IVN-{ARM}] DONE: {T} steps, {meta['wall_sec']:.0f}s, vram={meta['max_vram_gb']}, "
      f"累计衰减暴露={meta['cum_decay_exposure']:.3f}", flush=True)
