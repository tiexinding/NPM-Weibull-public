import re, json, numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr, kurtosis
DUMP = Path("<selfavg_mechanism>/10_p5batch_cloud/results/dump_early/tensors")
rng = np.random.default_rng(0)
def dc(Y):  # double centering -> residual
    return Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
rows = []
for st in steps:
    d = torch.load(DUMP / f"step{st}.pt", map_location="cpu", weights_only=False)
    for key in d:
        if key == "__meta__" or not key.endswith("|W"): continue
        base = key[:-2]; m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", base)
        L, kind = int(m.group(1)), m.group(2)
        W = d[base+"|W"].double().numpy(); V = d[base+"|Vh"].double().numpy()
        LV = np.log(V + 1e-300); eps = dc(LV)
        LW = np.log(np.abs(W) + 1e-300); lz = dc(LW)
        r2_eps = eps.var()/LV.var()
        e, z = eps.ravel(), lz.ravel()
        rho = spearmanr(e, z).correlation
        pear = np.corrcoef(e, z)[0,1]
        # shuffle null (within matrix)
        null = [spearmanr(e, rng.permutation(z)).correlation for _ in range(3)]
        # sign-aware: eps vs signed W? (should be ~0); eps vs |W| rank within row
        rows.append(dict(step=st, layer=L, kind=kind, r2_eps=r2_eps, eps_sd=eps.std(), eps_kurt=kurtosis(e),
                         rho=rho, pearson=pear, null_abs=float(np.abs(null).max())))
    print("done", st, flush=True)
json.dump(rows, open("eps_vs_z_rows.json","w"))
import collections
print(f"{'step':>6} {'kind':<10} {'R2eps':>6} {'sd':>5} {'kurt':>6} {'rho':>7} {'pear':>7} {'null':>6}")
for st in steps:
    for kind in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
        rr = [r for r in rows if r["step"]==st and r["kind"]==kind]
        med = lambda k: np.median([r[k] for r in rr])
        print(f"{st:>6} {kind:<10} {med('r2_eps'):6.3f} {med('eps_sd'):5.2f} {med('eps_kurt'):6.1f} {med('rho'):+7.3f} {med('pearson'):+7.3f} {max(r['null_abs'] for r in rr):6.3f}")
