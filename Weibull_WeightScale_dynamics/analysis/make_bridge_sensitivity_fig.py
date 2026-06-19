"""Bridge sensitivity figure. 3-panel:
(a) Gamma bridge ratio vs k (k-lock sensitivity) (b) sigma->lambda three curves (observed /
fixed-k / per-checkpoint-k) (c) closed-loop error over time.
"""
import json,numpy as np
from math import gamma,sqrt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
PF=""
def ratio(k): return 1/sqrt(gamma(1+2/k))
c=json.load(open("derived_data/real_pythia/realpythia_closure.json"))
step=np.array([x['step'] for x in c]); lo=np.array([x['lambda_obs'] for x in c])
so=np.array([x['sigma_obs'] for x in c]); lp=np.array([x['lambda_pred'] for x in c])
kby={int(s): v for s, v in json.load(open("derived_data/real_pythia/realpythia_k_by_step.json")).items()}
ks=np.array([kby.get(x['step'],1.16) for x in c])

fig=plt.figure(figsize=(15,4.3)); gs=gridspec.GridSpec(1,3,wspace=0.28)
# (a) Γ桥比 vs k
axa=fig.add_subplot(gs[0]); kk=np.linspace(1.14,1.24,100)
axa.plot(kk,[ratio(k) for k in kk],'-',c='#2c3e50',lw=2,label='bridge ratio $\\lambda/\\sigma=1/\\sqrt{\\Gamma(1+2/k)}$')
axa.axvline(1.20,ls='--',c='#27ae60',alpha=0.6,label='canonical $k$=1.20')
axa.axvspan(1.148,1.193,color='#f39c12',alpha=0.25,label='real Pythia $k(t)$ range')
axa.set_xlabel('Weibull shape $k$',fontsize=10.5); axa.set_ylabel('Bridge ratio $\\lambda/\\sigma=1/\\sqrt{\\Gamma(1+2/k)}$',fontsize=9.5)
axa.legend(fontsize=8.5); axa.set_title('(a) $k$-lock sensitivity: bridge varies $\\sim$2% over $k(t)$ range',fontsize=10)
# (b) σ→λ 3曲线
axb=fig.add_subplot(gs[1])
axb.plot(step,lo*1e3,'-o',ms=3,c='#c0392b',label='observed $\\lambda$ (Weibull fit)',zorder=5)
axb.plot(step,so*ratio(1.20)*1e3,'--s',ms=3,c='#2980b9',label='predicted: fixed $k$=1.20 (~4%)')
axb.plot(step,so*np.array([ratio(k) for k in ks])*1e3,':^',ms=3,c='#16a085',label='predicted: per-ckpt $k(t)$')
axb.set_xscale('log'); axb.set_xlabel('Training step',fontsize=10.5); axb.set_ylabel('$\\lambda$ ($\\times 10^{3}$)',fontsize=10.5)
axb.legend(fontsize=8.2); axb.set_title('(b) $\\sigma\\to\\lambda$ bridge: ~4% residual (Weibull-fit, not $k$)',fontsize=10)
# (c) closed-loop误差时间分布
axc=fig.add_subplot(gs[2])
br_err=100*np.abs(so*ratio(1.20)-lo)/lo; int_err=100*np.abs(lp/ratio(1.20)-so)/so; tot_err=100*np.abs(lp-lo)/lo
axc.axvspan(30000,1.6e5,color='#e74c3c',alpha=0.12,zorder=0)
axc.plot(step,tot_err,'-o',ms=3,c='#8e44ad',label='total closed-loop')
axc.plot(step,br_err,'--s',ms=3,c='#2980b9',label='bridge ($\\sigma\\to\\lambda$)')
axc.plot(step,int_err,':^',ms=3,c='#e67e22',label='integration (force$\\to\\sigma^2$)')
axc.set_xscale('log'); axc.set_xlabel('Training step',fontsize=10.5); axc.set_ylabel('Relative error (%)',fontsize=10.5)
axc.legend(fontsize=8.2); axc.set_title('(c) Error decomposition: dense region ~6% (bridge 4.6 + integ 1.9)',fontsize=9.5)
plt.savefig(PF+"F_bridge_sensitivity.png",dpi=150,bbox_inches="tight"); plt.close()
print("F_bridge_sensitivity.png ✓ (3-panel: k-lock敏感/σ→λ 3曲线/误差分解)")
print(f"  桥误差均值{np.nanmean(br_err):.1f}% / 积分{np.nanmean(int_err):.1f}% / 总{np.nanmean(tot_err):.1f}%")
