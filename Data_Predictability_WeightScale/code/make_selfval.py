#!/usr/bin/env python
"""Paper#3 自验证 (B2 6-18, 老丁): 仅凭数据 D 预测 λ vs 实际训练实测 λ, 盲测对比.
核心: 预测器输入 D = 训练前纯数据统计 → 主律 → λ_pred; 实训练 → 权重 Weibull → λ_obs.
L1 = 同语料留一(LOO)盲测(现实区间 ρ≤0.6); L2 = wiki 定系数→code/c4 跨语料盲测(验 §8 边界).
拆 A/B 两步(用真 struct_retain 定位误差在哪步). 全本地零云卡. 出 p3_out/p3_selfval.{html,png}.
"""
import numpy as np, os, json, glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

V = 50304
# ── 钉死 B 步常量 (凸版, §5: D = H_r − ΔH·S^1.69) ──
Hr, dH, PE = 8.0, 1.82, 1.69
Sinv = lambda D: np.power(np.clip((Hr - np.asarray(D)) / dH, 1e-9, None), 1.0 / PE)   # B 反推 S
def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev, nxt = t[:-1], t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))
def struct(f):   # struct_retain vs 干净 ρ0 (seq_len=512, 同 physics_fits 口径)
    a = np.load(f); n = (len(a) // 512) * 512
    o = np.load("tokens_corrupt/corrupt_rho000_s0.npy")[:n].reshape(-1, 512); c = a[:n].reshape(-1, 512)
    return float(((o[:, :-1] == c[:, :-1]) & (o[:, 1:] == c[:, 1:])).mean())
def dl2_of(jf):  # λ²−λ₀² (×1e4), + λ0
    j = json.load(open(jf)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4, j[0]["lambda"]
def R2(y, yh): y = np.asarray(y); return 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)

# ════════ 组装 L1 腐蚀族(wiki, η=3e-4, 现实区间 ρ≤0.6), 跨 seed 平均 ════════
TAGS = ["000", "010", "020", "030", "040", "050", "060"]; RHO = [int(t) / 100 for t in TAGS]
CROSS = {"000", "020", "040", "060"}   # 这些 ρ 有 s1,s2 (Phase B)
D1, S1, Y1, Yerr, N1 = [], [], [], [], []
for tg in TAGS:
    fs = [f"_tmp_corrupt/corrupt_rho{tg}_s0/lambda_trajectory_continue.json"]
    if tg in CROSS: fs += [f"_tmp_grid/rho{tg}_e3e-4_s{s}/lambda_trajectory_continue.json" for s in (1, 2)]
    ys = [dl2_of(f)[0] for f in fs if os.path.exists(f)]
    D1.append(D_bi(np.load(f"tokens_corrupt/corrupt_rho{tg}_s0.npy")))
    S1.append(struct(f"tokens_corrupt/corrupt_rho{tg}_s0.npy"))
    Y1.append(np.mean(ys)); Yerr.append(np.std(ys) if len(ys) > 1 else 0.0); N1.append(len(ys))
D1, S1, Y1, Yerr = map(np.array, [D1, S1, Y1, Yerr])

# ════════ L1: 留一(LOO)盲测 + 拆 A/B ════════
def fitA(Sx, Yx):  # A 步: λ²−λ₀² = C0 + C1·S
    C1, C0 = np.polyfit(Sx, Yx, 1); return C0, C1
# 全数据 in-sample
C0a, C1a = fitA(S1, Y1); Yins = C0a + C1a * S1
loo_full, loo_A, Dpred_B = [], [], []
for i in range(len(TAGS)):
    m = np.arange(len(TAGS)) != i
    C0, C1 = fitA(S1[m], Y1[m])
    loo_full.append(C0 + C1 * Sinv(D1[i]))    # 全链: 仅从 D 预测 (B 反推 S → A)
    loo_A.append(C0 + C1 * S1[i])             # 只 A: 用真 S
    Dpred_B.append(Hr - dH * S1[i]**PE)       # 只 B: 用真 S 预测 D
loo_full, loo_A, Dpred_B = map(np.array, [loo_full, loo_A, Dpred_B])
rmse_ins = float(np.sqrt(np.mean((Y1 - Yins)**2)))
rmse_loo = float(np.sqrt(np.mean((Y1 - loo_full)**2)))
rmse_Aonly = float(np.sqrt(np.mean((Y1 - loo_A)**2)))     # A 步误差(给真 S)
rmse_B = float(np.sqrt(np.mean((D1 - Dpred_B)**2)))       # B 步误差(D 单位)
relerr_loo = float(np.mean(np.abs(Y1 - loo_full) / np.abs(Y1)))

# ════════ L2: 全 wiki 腐蚀族定系数 → 自然语料 code/c4/wiki 盲测 ════════
NAT = {"wikitext": "tokens_corrupt/corrupt_rho000_s0.npy", "c4": "tokens/c4_30M.npy", "code": "tokens/code_30M.npy"}
def nat_y(name):
    fs = [f"L3_results_e3e4/continue_{name}_s{s}/lambda_trajectory_continue.json" for s in (0, 1, 2)]
    ys = [dl2_of(f)[0] for f in fs if os.path.exists(f)]
    if not ys:  # 退回 p3_out/continue_*
        f = f"p3_out/continue_{name}/lambda_trajectory_continue.json"
        if os.path.exists(f): ys = [dl2_of(f)[0]]
    return (np.mean(ys), np.std(ys) if len(ys) > 1 else 0.0, len(ys)) if ys else (None, 0, 0)
C0all, C1all = C0a, C1a   # wiki 腐蚀族全量系数
L2 = []
for name, tk in NAT.items():
    Dn = D_bi(np.load(tk)); yo, ye, nn = nat_y(name)
    if yo is None: continue
    yp = C0all + C1all * Sinv(Dn)          # 全链从 D 预测(自然语料无真 S → 必用 D)
    L2.append((name, Dn, yo, ye, nn, yp, yo - yp))

# ════════ 图: 预测 vs 实测 ════════
fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44],
                    subplot_titles=("(A) 预测 vs 实测 λ²−λ₀²  (对角线=完美)",
                                    "(B) 误差拆解:A 步(真S) vs 全链(仅D)"))
lo, hi = 0.5, 2.6
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#999", dash="dash"), name="y=x 完美预测"), row=1, col=1)
fig.add_trace(go.Scatter(x=Y1, y=loo_full, mode="markers", marker=dict(color="#2c5aa0", size=11, symbol="circle"),
                         error_x=dict(type="data", array=Yerr, visible=True, color="#9bb"),
                         name=f"L1 同语料 LOO 盲测 (n_seed≤3)"), row=1, col=1)
for name, Dn, yo, ye, nn, yp, err in L2:
    col = {"code": "#d62728", "c4": "#2a9d5a", "wikitext": "#7f7f7f"}[name]
    fig.add_trace(go.Scatter(x=[yo], y=[yp], mode="markers+text", marker=dict(color=col, size=15, symbol="star"),
                             text=[f" {name}"], textposition="middle right", textfont=dict(size=11),
                             name=f"L2 {name} 盲测 (Δ={err:+.2f})"), row=1, col=1)
fig.update_xaxes(title_text="实测 λ²−λ₀² (×10⁻⁴)", range=[lo, hi], row=1, col=1)
fig.update_yaxes(title_text="预测 λ²−λ₀² (×10⁻⁴)", range=[lo, hi], row=1, col=1)
# B 面板: 每个 ρ 的 A 步误差 vs 全链误差
fig.add_trace(go.Bar(x=[f"ρ={r:.1f}" for r in RHO], y=np.abs(Y1 - loo_A), name="A 步误差(给真S)", marker_color="#9467bd"), row=1, col=2)
fig.add_trace(go.Bar(x=[f"ρ={r:.1f}" for r in RHO], y=np.abs(Y1 - loo_full), name="全链误差(仅D)", marker_color="#2c5aa0"), row=1, col=2)
fig.update_yaxes(title_text="|误差| (×10⁻⁴)", row=1, col=2)
fig.update_layout(template="plotly_white", height=520, width=1080, barmode="group",
                  legend=dict(orientation="h", y=-0.18),
                  title_text=f"Paper#3 自验证:仅凭数据 D 预测 λ vs 实训练实测 (L1 LOO RMSE={rmse_loo:.3f} / 相对 {relerr_loo*100:.1f}%)")
os.makedirs("p3_out", exist_ok=True)
fig.write_html("p3_out/p3_selfval.html", include_plotlyjs="inline")
try: fig.write_image("p3_out/p3_selfval.png", scale=2)
except Exception as e: print("png skip", e)

# ════════ 打印结果表 ════════
print("WROTE p3_out/p3_selfval.{html,png}\n")
print("=== B 步独立检验 (D=H_r−ΔH·S^1.69, 钉死) ===")
print(f"  D R²(真S→D)={R2(D1, Dpred_B):.4f}  RMSE_B={rmse_B:.3f} bits  → B 步{'稳' if R2(D1,Dpred_B)>0.99 else '需查'}")
print("=== L1 同语料 LOO 盲测 (现实区间 ρ≤0.6, η=3e-4) ===")
print(f"  {'ρ':>4} {'n_seed':>6} {'D':>6} {'S_true':>7} {'λ²obs':>7} {'λ²pred(LOO)':>11} {'err':>7} {'A步err':>7}")
for i, tg in enumerate(TAGS):
    print(f"  {RHO[i]:>4.1f} {N1[i]:>6} {D1[i]:>6.2f} {S1[i]:>7.3f} {Y1[i]:>7.3f} {loo_full[i]:>11.3f} {Y1[i]-loo_full[i]:>+7.3f} {Y1[i]-loo_A[i]:>+7.3f}")
print(f"  in-sample RMSE={rmse_ins:.3f}  |  LOO RMSE={rmse_loo:.3f} (相对 {relerr_loo*100:.1f}%)  |  过拟合比={rmse_loo/max(rmse_ins,1e-9):.2f}×")
print(f"  A 步误差(给真S) RMSE={rmse_Aonly:.3f}  → 全链 vs A 步差 = B 反推注入的误差")
print("=== L2 跨语料盲测 (wiki 腐蚀族系数 C0=%.2f C1=%.2f → 预测自然语料) ===" % (C0all, C1all))
print(f"  {'语料':>10} {'n':>3} {'D':>6} {'λ²obs':>7} {'λ²pred':>7} {'err(obs−pred)':>13}")
for name, Dn, yo, ye, nn, yp, err in L2:
    flag = " ⟵ 偏离(冗余维 §8)" if abs(err) > 3 * rmse_loo else ""
    print(f"  {name:>10} {nn:>3} {Dn:>6.2f} {yo:>7.3f} {yp:>7.3f} {err:>+13.3f}{flag}")
