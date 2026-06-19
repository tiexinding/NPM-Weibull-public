#!/usr/bin/env python
"""Three-force decomposition + spline displacement recovery (Paper #2).

Self-trains a transformer (Pythia/Llama architecture, random init) and logs, at each
recorded step, the leading-order three-force budget of the squared weight norm
(alignment / injection / decay) using ground-truth AdamW moments, together with
per-layer force shares and per-layer Weibull k (middle-80% probability-plot protocol,
Paper #1).

Spline displacement recovery: build a cubic spline over checkpoint spacings
{250, 500, 1000}, then estimate the alignment force from sparse checkpoints as
  u_hat = -(cs(t+1) - cs(t)) / eta - lambda_wd * cs(t),
and compare spline-recovered vs ground-truth alignment (a single two-point baseline is
logged alongside).

Offline by design: no HuggingFace model / dataset / tokenizer download. The model is
constructed from a local config (pythia70m_config.json); the corpus is a pre-tokenized
.npy of public wikitext-103 (see pretok_local.py). Results are written incrementally to
<out>/v1b_spline_true.jsonl.

Usage:
  HF_HUB_OFFLINE=1 python three_force_spline.py --arch pythia --steps 20000 --rec 250 \
      --tokens data/wikitext_pythia_tokens.npy --config pythia70m_config.json --out runs/pythia
"""
import os, math, json, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
from scipy.interpolate import CubicSpline
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--rec", type=int, default=250)
ap.add_argument("--sub", type=int, default=200000)
ap.add_argument("--tokens", default="data/wikitext_pythia_tokens.npy")
ap.add_argument("--config", default="pythia70m_config.json")
ap.add_argument("--arch", default="pythia", choices=["pythia", "llama"])
ap.add_argument("--include_vproj", action="store_true", help="Llama: 含 v_proj(对照); 默认排除以对齐 Pythia o+FFN(B1 catch 可比性)")
ap.add_argument("--out", default="runs/three_force")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--spacings", default="250,500,1000")  # 干跑可传小间隔触发样条路径
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--warm", type=int, default=200)
ap.add_argument("--save_ckpt", action="store_true", help="存全模型权重 ckpt (log+linear 调度, weights-only) → 可复用 dense-ckpt 资源")
ap.add_argument("--ckpt_dir", default="")
ap.add_argument("--lwd", type=float, default=0.01, help="weight decay λwd; 调大(如0.05)缩短 τ_iter=1/(η·λwd) → 20k步内 T/τ→1 看 relax")
ap.add_argument("--eta", type=float, default=1e-3, help="peak learning rate")
ap.add_argument("--beta2", type=float, default=0.999, help="AdamW β2 (真Pythia=0.95, 默认我0.999); 隔离峰值2×")
ap.add_argument("--init_from", default="", help="从真模型权重init(continue-train), 如 pythia-70m; 空=随机")
ap.add_argument("--init_rev", default="main", help="init权重revision: main(=step143000终态) 或 stepN")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
ETA = a.eta; LWD = a.lwd; B1 = 0.9; B2 = a.beta2; EPS = 1e-8; T = a.steps; WARM = a.warm; EMIN = 0.1; BS = a.bs; SEQ = a.seq
SPACINGS = [int(s) for s in a.spacings.split(",")]

def lr(t):
    if t < WARM: return ETA * t / WARM
    f = (t - WARM) / (T - WARM); return ETA * EMIN + 0.5 * ETA * (1 - EMIN) * (1 + math.cos(math.pi * f))

# ---- 零下载: 离线 config 构造模型 (arch 切换, Pythia 默认不变) ----
if a.arch == "llama":
    cfg = LlamaConfig.from_json_file(a.config); model = LlamaForCausalLM(cfg).to(dev)
else:
    cfg = GPTNeoXConfig.from_json_file(a.config); model = GPTNeoXForCausalLM(cfg).to(dev)
# ---- optional: initialize from real Pythia weights (continue-training control) ----
if getattr(a, "init_from", ""):
    import subprocess as _sp
    binf = f"{a.out}/_init.bin"; os.makedirs(a.out, exist_ok=True)
    rev = a.init_rev  # e.g. main(=step143000) 或 step1
    url = f"https://hf-mirror.com/EleutherAI/{a.init_from}/resolve/{rev}/pytorch_model.bin"
    print(f"[init] 下真Pythia权重 {a.init_from}@{rev}", flush=True)
    _sp.run(["curl","-L","--connect-timeout","20","-m","200","--retry","3","-C","-","-s","-o",binf,url])
    sd = torch.load(binf, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[init] loaded 真Pythia权重 (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    os.remove(binf)
model.train()

# ---- 零下载: 预 tokenize 语料 ----
toks = np.load(a.tokens).astype(np.int64)
rng = np.random.default_rng(a.seed)
def batch():
    idx = rng.integers(0, len(toks) - SEQ - 1, BS)
    return torch.tensor(np.stack([toks[i:i + SEQ] for i in idx]), dtype=torch.long, device=dev)

# Llama 默认排 v_proj 对齐 Pythia(o+FFN), 只变架构=可比性优先 (B1 6-7 catch); --include_vproj 跑对照
_LLAMA_TRANS = (["v_proj"] if a.include_vproj else []) + ["o_proj", "gate_proj", "up_proj", "down_proj"]
_TRANS = {"pythia": ["attention.dense", "dense_h_to_4h", "dense_4h_to_h"], "llama": _LLAMA_TRANS}[a.arch]
def is_trans(n): return any(s in n for s in _TRANS) and n.endswith(".weight")
# B-sel (审稿1 6.2.5): Selection class(Q/K)三力. Llama q_proj/k_proj分开干净; Pythia query_key_value融合(含V, caveat)
_SELECT = {"pythia": ["query_key_value"], "llama": ["q_proj", "k_proj"]}[a.arch]
def is_select(n): return any(s in n for s in _SELECT) and n.endswith(".weight")
def layer_id(n):  # gpt_neox.layers.<L>.<...>  → L
    for p in n.split("."):
        if p.isdigit(): return int(p)
    return -1

def weibull_k80(x):
    """Paper1 中段80%概率图协议: 对 |w| 排序, F=(i-0.5)/n, 保留 F∈[0.1,0.9], 拟合 log(-log(1-F)) vs log x, 斜率=k。"""
    x = np.sort(np.abs(x)); x = x[x > 0]; n = len(x)
    if n < 50: return float("nan")
    F = (np.arange(1, n + 1) - 0.5) / n
    m = (F >= 0.1) & (F <= 0.9)
    if m.sum() < 20: return float("nan")
    y = np.log(-np.log(1 - F[m])); lx = np.log(x[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    k, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(k)

opt = torch.optim.AdamW(model.parameters(), lr=ETA, weight_decay=LWD, betas=(B1, B2), eps=EPS)
trans = [n for n, p in model.named_parameters() if is_trans(n)]; pmap = dict(model.named_parameters())
sel = [n for n, p in model.named_parameters() if is_select(n)]  # B-sel: Selection(Q/K)层
print(f"[spline] dev={dev} transmission层数={len(trans)} 总参={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

# 固定子采样索引 (每层按占比, 合计 ~sub) — 每快照同一子集
sizes = {nm: pmap[nm].numel() for nm in trans}; tot = sum(sizes.values())
subidx = {nm: torch.from_numpy(np.sort(rng.choice(sizes[nm], max(1, int(a.sub * sizes[nm] / tot)), replace=False))).to(dev) for nm in trans}
def gather_W():
    return torch.cat([pmap[nm].data.flatten()[subidx[nm]].double().cpu() for nm in trans]).numpy()

REC = sorted(set(range(a.rec, T + 1, a.rec)))
Wsub = {}; true_al_sub = {}
tf_path = f"{a.out}/v1b_spline_true.jsonl"; open(tf_path, "w").close()
pl_path = f"{a.out}/v1b_perlayer_budget.jsonl"; open(pl_path, "w").close()

# 可复用 dense-ckpt 资源 (weights-only, log+linear 调度模仿 Pythia, 含真模型缺的 log-dense 早期段)
CKPT_DIR = a.ckpt_dir or f"{a.out}/ckpts"
CKPT_STEPS = sorted(set([1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, T + 1, 1000))))
if a.save_ckpt:
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[ckpt] 将存 {len([s for s in CKPT_STEPS if s<=T])} 个 weights-only ckpt → {CKPT_DIR}", flush=True)

for t in range(1, T + 1):
    cur = lr(t)
    for g in opt.param_groups: g['lr'] = cur
    opt.zero_grad(set_to_none=True); x = batch(); out = model(input_ids=x, labels=x); out.loss.backward(); opt.step()
    if a.save_ckpt and t in CKPT_STEPS:
        torch.save({k: v.detach().to(torch.float32).cpu() for k, v in model.state_dict().items()}, f"{CKPT_DIR}/step{t}.pt")
        print(f"[ckpt] saved step{t}", flush=True)
    if t in REC:
        al_s = 0.0; al_f = inj_f = dec_f = sw = 0.0; n_f = 0
        per_layer = {}  # T3: per-layer budget + k
        for nm in trans:
            p = pmap[nm]; st = opt.state[p]
            mh = st['exp_avg'] / (1 - B1 ** t); vh = st['exp_avg_sq'] / (1 - B2 ** t); u = mh / (vh.sqrt() + EPS)
            uf = u.flatten(); pf = p.data.flatten()
            al_layer = float((p.data * u).sum()); inj_layer = float((u * u).sum()); dec_layer = float((p.data ** 2).sum())
            al_s += float((pf[subidx[nm]] * uf[subidx[nm]]).sum())   # 子集对齐 (公平对比样条)
            al_f += al_layer; inj_f += inj_layer; dec_f += dec_layer; sw += dec_layer; n_f += p.numel()
            wsamp = pf[subidx[nm]].detach().cpu().numpy()
            per_layer[nm] = {"layer": layer_id(nm),
                             "align": 2 * cur * al_layer, "inj": cur * cur * inj_layer, "dec": 2 * cur * LWD * dec_layer,
                             "rms": math.sqrt(dec_layer / p.numel()), "k80": weibull_k80(wsamp)}
        # B-sel: Selection(Q/K)三力 (与transmission同法, 比较力平衡)
        al_sel = inj_sel = dec_sel = sw_sel = 0.0; n_sel = 0
        for nm in sel:
            p = pmap[nm]; st = opt.state[p]
            mh = st['exp_avg'] / (1 - B1 ** t); vh = st['exp_avg_sq'] / (1 - B2 ** t); u = mh / (vh.sqrt() + EPS)
            al_sel += float((p.data * u).sum()); inj_sel += float((u * u).sum())
            dec_sel += float((p.data ** 2).sum()); sw_sel += float((p.data ** 2).sum()); n_sel += p.numel()
        true_al_sub[t] = 2 * cur * al_s; Wsub[t] = gather_W()
        rec = {"step": t, "eta": cur, "true_align_sub": 2 * cur * al_s, "true_align_full": 2 * cur * al_f,
               "inj_full": cur * cur * inj_f, "dec_full": 2 * cur * LWD * dec_f, "rms": math.sqrt(sw / n_f),
               "sel_align": 2 * cur * al_sel, "sel_inj": cur * cur * inj_sel, "sel_dec": 2 * cur * LWD * dec_sel,
               "sel_rms": math.sqrt(sw_sel / n_sel) if n_sel else 0.0}
        with open(tf_path, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush()
        with open(pl_path, "a") as f: f.write(json.dumps({"step": t, "eta": cur, "layers": per_layer}) + "\n"); f.flush()
        # net force budget 占比 (绝对值归一, 给 P2 force-budget 解读)
        a_, i_, d_ = abs(rec["true_align_full"]), rec["inj_full"], rec["dec_full"]; tt = a_ + i_ + d_ + 1e-30
        print(f"[spline] step{t} rms={rec['rms']:.4f} align%={100*a_/tt:.1f} inj%={100*i_/tt:.1f} dec%={100*d_/tt:.1f}", flush=True)

# ===== 训完: 三间隔样条 + 2点 对照 (子集) =====
steps_fine = sorted(Wsub); Wmat_all = {s: Wsub[s] for s in steps_fine}
ev = [t for t in steps_fine if t in true_al_sub]
out_res = {}
for S in SPACINGS:
    sps = [s for s in steps_fine if s % S == 0]
    if len(sps) < 4: continue
    Wmat = np.array([Wmat_all[s] for s in sps]); cs = CubicSpline(sps, Wmat, axis=0)
    rs_sp = []; rs_2p = []; rows = []
    for t in ev:
        if t < min(sps) or t >= max(sps): continue
        et = lr(t); Wt = cs(t); Wt1 = cs(t + 1)
        u_sp = -(Wt1 - Wt) / et - LWD * Wt; al_sp = 2 * et * float(np.sum(Wt * u_sp))
        lo = (t // S) * S; hi = lo + S; al_2p = float('nan')
        if lo in Wmat_all and hi in Wmat_all:
            ig = 0.5 * (lr(lo) + lr(hi)) * (hi - lo); u2 = -(Wmat_all[hi] - Wmat_all[lo]) / ig - LWD * Wmat_all[lo]
            al_2p = 2 * et * float(np.sum(Wmat_all[lo] * u2))
        ta = true_al_sub[t]
        rows.append({"step": t, "true": ta, "spline": al_sp, "2pt": al_2p,
                     "spline_ratio": al_sp / ta if ta else None, "2pt_ratio": al_2p / ta if ta else None})
        if ta: rs_sp.append(al_sp / ta); rs_2p.append(al_2p / ta if not math.isnan(al_2p) else np.nan)
    med_sp = float(np.nanmedian(rs_sp)) if rs_sp else None; med_2p = float(np.nanmedian(rs_2p)) if rs_2p else None
    out_res[str(S)] = {"median_spline_ratio": med_sp, "median_2pt_ratio": med_2p, "rows": rows}
    json.dump(out_res, open(f"{a.out}/v1b_spline_verify.json", "w"))
    print(f"[spline] === S={S}: 样条 align ratio 中位={med_sp} | 2点 中位={med_2p} (n={len(rs_sp)}) ===", flush=True)
print("[spline] DONE. 判据: 样条 ratio→1(≤~10%) 则A样条法真模型站住; 远离1则限定密ckpt。", flush=True)
