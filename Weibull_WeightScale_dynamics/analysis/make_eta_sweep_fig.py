"""η sweep 图. 2-panel: (a)peak λ vs η(log-log, ∝η^0.78 vs √η floor) (b)rise align%跨η稳定.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
PF="paper_figures/"
eta=np.array([3e-4,1e-3,3e-3]); peak=np.array([0.0313,0.0763,0.1874])
al_lo=np.array([85.1,88.3,82.4]); al_hi=np.array([93.7,93.8,93.9])  # rise align% range
fig=plt.figure(figsize=(11,4.3)); gs=gridspec.GridSpec(1,2,wspace=0.27)

# (a) peak λ vs η: data ∝η^0.78 vs √η floor
axa=fig.add_subplot(gs[0])
ee=np.logspace(np.log10(2e-4),np.log10(4e-3),50)
floor=peak[1]*np.sqrt(ee/1e-3)        # √(η/λwd) floor, 归一1e-3
fit=peak[1]*(ee/1e-3)**0.78            # peak拟合
axa.loglog(ee,floor,'--',c='#7f8c8d',lw=1.8,label='equilibrium floor $\\propto\\sqrt{\\eta/\\lambda_{wd}}$ ($\\eta^{0.5}$)')
axa.loglog(ee,fit,'-',c='#d95f0e',lw=1.5,alpha=0.6,label='peak fit $\\propto\\eta^{0.78}$')
axa.loglog(eta,peak,'o',c='#c0392b',ms=10,zorder=5,label='measured peak $\\lambda$')
for e,p in zip(eta,peak): axa.annotate(f'{p:.3f}',(e,p),textcoords='offset points',xytext=(6,6),fontsize=9,fontweight='bold')
axa.set_xlabel('Learning rate $\\eta$',fontsize=11); axa.set_ylabel('Peak Weibull $\\lambda$',fontsize=11)
axa.legend(fontsize=8.5,loc='upper left'); axa.set_title('(a) Peak $\\lambda$ scales steeper than the $\\sqrt{\\eta/\\lambda_{wd}}$ floor',fontsize=10)
# 加刻度 (横纵都多标几个)
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
axa.xaxis.set_major_locator(FixedLocator([3e-4,5e-4,1e-3,2e-3,3e-3])); axa.xaxis.set_minor_locator(NullLocator())
axa.xaxis.set_major_formatter(FixedFormatter(['3e-4','5e-4','1e-3','2e-3','3e-3']))
axa.yaxis.set_major_locator(FixedLocator([0.03,0.05,0.08,0.12,0.19])); axa.yaxis.set_minor_locator(NullLocator())
axa.yaxis.set_major_formatter(FixedFormatter(['0.03','0.05','0.08','0.12','0.19']))
axa.tick_params(labelsize=8.5)

# (b) rise align% vs η
axb=fig.add_subplot(gs[1])
x=np.arange(3); mid=(al_lo+al_hi)/2
axb.errorbar(x,mid,yerr=[mid-al_lo,al_hi-mid],fmt='s',ms=9,c='#2c7fb8',capsize=6,lw=2,label='rise-phase align% range')
axb.axhspan(88,94,color='#7fcdbb',alpha=0.25,label='baseline 88-94%')
axb.set_xticks(x); axb.set_xticklabels(['$3\\times10^{-4}$','$10^{-3}$','$3\\times10^{-3}$'])
axb.set_xlabel('Learning rate $\\eta$',fontsize=11); axb.set_ylabel('Rise-phase alignment share (%)',fontsize=11)
axb.set_ylim(75,100); axb.legend(fontsize=8.5,loc='lower center')
axb.set_title('(b) Alignment dominance is $\\eta$-robust ($\\sim$85-94% across $\\eta$)',fontsize=10)
plt.savefig(PF+"F_eta_sweep.png",dpi=150,bbox_inches="tight"); plt.close()
print("F_eta_sweep.png ✓ (a: peak∝η^0.78 vs √η floor / b: align% η-robust)")
