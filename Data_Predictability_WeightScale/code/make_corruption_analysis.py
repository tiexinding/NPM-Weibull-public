#!/usr/bin/env python
"""Paper#3 v2 实验1 判读: λ-vs-ρ + ∫F_align-vs-ρ + 单调性/动态范围/LOO R² (B2 6-16).
读 _tmp_corrupt/corrupt_rho*_s*/ 的 lambda_trajectory_continue.json + v1b_perlayer_budget.jsonl.
裁决: 强信号 = 单调 + 大range(>> c4-vs-wiki ~0.003) + LOO R² 高 → Paper#3 主结果立住.
用法: python make_corruption_analysis.py [_tmp_corrupt]
"""
import sys, os, glob, json
import numpy as np


def load_lambda_end(d):
    f = os.path.join(d, "lambda_trajectory_continue.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    # 兼容: 取末端 λ (字段名可能是 lambda/lam/traj)
    for k in ("lambda", "lam", "lambdas", "traj", "trajectory"):
        if k in j and isinstance(j[k], list) and j[k]:
            return float(j[k][-1])
    if isinstance(j, list) and j:
        last = j[-1]
        return float(last.get("lambda", last.get("lam")) if isinstance(last, dict) else last)
    return None


def load_intF(d):
    f = os.path.join(d, "v1b_perlayer_budget.jsonl")
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    steps = np.array([r["step"] for r in rows])
    post = steps >= 300
    tot = 0.0
    for b in rows[0]["layers"]:
        al = np.array([r["layers"][b]["align"] for r in rows])
        ap = al[post]; sp = steps[post]
        tot += sum(abs(ap[i]) * (sp[i] - sp[i - 1]) for i in range(1, len(sp)))
    return float(tot)


def max_align(d):
    f = os.path.join(d, "v1b_perlayer_budget.jsonl")
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    return float(max(abs(r["layers"][b]["align"]) for r in rows for b in r["layers"]))


def loo_r2(x, y):
    """留一线性拟合 R² (x=ρ, y=λ)."""
    x = np.array(x); y = np.array(y); n = len(x)
    if n < 4:
        return None
    pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        pred[i] = a * x[i] + b
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None


def main(root="_tmp_corrupt"):
    dirs = sorted(glob.glob(os.path.join(root, "corrupt_rho*_s*")))
    rows = []
    for d in dirs:
        base = os.path.basename(d)
        rho = int(base.split("rho")[1].split("_")[0]) / 100
        seed = base.split("_s")[1]
        rows.append(dict(rho=rho, seed=seed, lam=load_lambda_end(d),
                         intF=load_intF(d), maxalign=max_align(d), dir=base))
    rows = [r for r in rows if r["lam"] is not None]
    rows.sort(key=lambda r: (r["rho"], r["seed"]))
    print(f"=== 腐蚀梯度判读 ({len(rows)} runs) ===")
    print(f"{'ρ':>5} {'seed':>4} {'λ_end':>9} {'∫F_align':>10} {'max_align(守#84)':>15}")
    for r in rows:
        spike = "⚠️SPIKE" if (r["maxalign"] or 0) > 10 else ""
        print(f"{r['rho']:>5.2f} {r['seed']:>4} {r['lam']:>9.5f} "
              f"{(r['intF'] or 0):>10.2f} {(r['maxalign'] or 0):>14.2f} {spike}")
    # 聚合(多seed取均值)
    rhos = sorted(set(r["rho"] for r in rows))
    lam_by = {ro: [r["lam"] for r in rows if r["rho"] == ro] for ro in rhos}
    intF_by = {ro: [r["intF"] for r in rows if r["rho"] == ro and r["intF"]] for ro in rhos}
    x = rhos
    ylam = [np.mean(lam_by[ro]) for ro in rhos]
    yint = [np.mean(intF_by[ro]) if intF_by[ro] else np.nan for ro in rhos]
    # 裁决指标
    rng = max(ylam) - min(ylam)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho_sp = spearman(x, ylam)
    r2 = loo_r2(x, ylam)
    print("\n=== 裁决 ===")
    print(f"λ 动态范围(max-min)  : {rng:.5f}   (对比 c4-vs-wiki ~0.003: {'✅大' if rng > 0.006 else '⚠️不大'})")
    print(f"单调性 Spearman(ρ,λ) : {rho_sp:+.3f}   ({'✅单调' if abs(rho_sp) > 0.85 else '⚠️弱'})")
    print(f"LOO 线性 R²          : {r2 if r2 is None else round(r2,3)}   ({'✅高' if (r2 or 0) > 0.8 else '⚠️低'})")
    verdict = (rng > 0.006 and abs(rho_sp) > 0.85 and (r2 or 0) > 0.8)
    print(f"\n{'🟢 强信号 → Paper#3 主结果立住' if verdict else '🔴 信号不够干净 → 诚实回头(守#80)'}")
    # 存 json 供画图
    json.dump(dict(rhos=x, lam=ylam, intF=yint, range=rng, spearman=rho_sp, loo_r2=r2,
                   verdict=bool(verdict), n_runs=len(rows)),
              open(os.path.join(root, "_corruption_verdict.json"), "w"), indent=2)
    print(f"\n[saved] {root}/_corruption_verdict.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_tmp_corrupt")
