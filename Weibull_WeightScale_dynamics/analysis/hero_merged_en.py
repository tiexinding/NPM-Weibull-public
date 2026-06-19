"""C1 HERO 英文版. 三力拔河推λ + 真实λ(t)四阶段时间线 + 主方程.
英文 + 数字85-91%→88-94%(v12) + paper适配. mathtext渲染û/m̂/v̂."""
import json,glob,re,math,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.interpolate import PchipInterpolator
RED,BLUE,GREEN='#c0392b','#2980b9','#27ae60'
# 数据源: realpythia_closure (aggregate transmission λ_obs) — 与§6/F8/MASTER一致(峰0.0352@30k), 
RP=json.load(open('derived_data/real_pythia/realpythia_closure.json'))
pc={x['step']:x['lambda_obs'] for x in RP if x['step']>=1}
_st=[s for s in sorted(pc) if s>=1]; steps=np.array(_st); obs=np.array([pc[s] for s in _st])*1e3
sx=np.clip(steps,1,None); xs=np.logspace(0,math.log10(sx.max()),400)
ys=PchipInterpolator(np.log10(sx),obs)(np.log10(xs))
T=143000; WARM=1430; EMIN=0.1; EP=1e-3; LWD=0.01
def eta(t):
    if t<WARM: return EP*t/WARM
    f=(t-WARM)/(T-WARM); return EP*EMIN+0.5*EP*(1-EMIN)*(1+math.cos(math.pi*f))
tgt=np.array([math.sqrt(eta(max(x,1))/LWD) for x in xs]); tgt=tgt/tgt.max()*obs.max()

fig=plt.figure(figsize=(16.5,9.4))
gs=fig.add_gridspec(2,2,height_ratios=[3.0,1.35],width_ratios=[1,1.72],hspace=0.30,wspace=0.14)
axL=fig.add_subplot(gs[0,0]); axR=fig.add_subplot(gs[0,1]); axk=fig.add_subplot(gs[1,:]); axk.axis('off')

# ===== 左: three-force tug-of-war =====
axL.set_xlim(0,10); axL.set_ylim(-0.3,10.3); axL.axis('off')
axL.set_title('(a) Three-force balance on $\\lambda$',fontsize=15)
axL.add_patch(Circle((5,5),1.4,fc='#f4d03f',ec='#b7950b',lw=2.5,zorder=3))
axL.text(5,5,'Weight\nscale $\\lambda$',ha='center',va='center',fontsize=14,zorder=4)
axL.add_patch(FancyArrowPatch((5,6.55),(5,9.0),arrowstyle='-|>',mutation_scale=27,lw=4.6,color=RED,zorder=2))
axL.text(5,9.6,'Alignment (learning)',ha='center',fontsize=13.5,color=RED)
axL.text(6.6,7.85,'pulls up\ndominant',ha='left',fontsize=11.5,color=RED)
axL.add_patch(FancyArrowPatch((3.75,6.1),(2.1,7.4),arrowstyle='-|>',mutation_scale=15,lw=2.3,color=BLUE,zorder=2))
axL.text(1.7,7.9,'Injection\n(noise)',ha='center',fontsize=11.5,color=BLUE)
axL.add_patch(FancyArrowPatch((5,3.45),(5,1.0),arrowstyle='-|>',mutation_scale=27,lw=4.6,color=GREEN,zorder=2))
axL.text(5,0.4,'Decay (weight decay)',ha='center',fontsize=12.5,color='#1e8449')
axL.text(6.6,2.4,'pulls down',ha='left',fontsize=11.5,color='#1e8449')

# ===== 右: real λ(t) mechanism timeline =====
# 阶段边界对齐 aggregate 峰@30k
bands=[(1,1500,'#eef2f3','#5d6d7e','1'),(1500,30000,'#fdebd0','#b9770e','2'),
       (30000,55000,'#fadbd8','#a93226','3'),(55000,150000,'#eafaf1','#1e8449','4')]
for a,b,c,_,_ in bands: axR.axvspan(a,b,color=c,alpha=0.95,zorder=0)
axR.plot(xs,tgt,'--',color='#7f8c8d',lw=2.0,label='Moving target $\\lambda^*=\\sqrt{\\eta/\\lambda_{wd}}$',zorder=2)
axR.plot(xs,ys,'-',color=RED,lw=2.0,alpha=0.5,zorder=3)
axR.plot(sx,obs,'o',color=RED,ms=7,zorder=4,label='Real $\\lambda(t)$ (Pythia-70m, transmission)')
axR.set_xscale('log'); axR.set_ylim(0,52); axR.set_xlim(1,1.6e5)
axR.set_ylabel('Weight scale $\\lambda$ ($\\times 0.001$)',fontsize=12.5)
axR.set_title('(b) Real $\\lambda(t)$ mechanism timeline (dominant force per phase)',fontsize=15)
axr=axR.twinx(); axr.set_yscale('log'); axr.set_ylim(1e-5,1.3e-1)
etac=np.array([eta(max(x,1)) for x in xs])
axr.plot(xs,etac,'-',color='#8d6e63',lw=1.9,zorder=2,label='Learning rate $\\eta(t)$: warmup$\\to$cosine')
axr.axhline(LWD,ls='-.',color='#7d3c98',lw=1.4,zorder=1,label='$\\lambda_{wd}=0.01$ (fixed)')
axr.set_ylabel('$\\eta$, $\\lambda_{wd}$ (log)',fontsize=11.5,color='#6d4c41'); axr.tick_params(axis='y',labelcolor='#6d4c41')
axr.annotate('$\\eta$ peaks @1430 (cosine decay after)',xy=(1430,1e-3),xytext=(2400,4e-3),fontsize=9.5,color='#6d4c41',
             arrowprops=dict(arrowstyle='->',color='#6d4c41',lw=1))
for a,b,_,ec,n in bands:
    xm=10**((math.log10(a)+math.log10(min(b,1.5e5)))/2)
    axR.text(xm,1.7,n,ha='center',va='bottom',fontsize=16,color=ec,zorder=5,fontweight='bold')
axR.set_xlabel('Training step (log)',fontsize=12.5)
h1,l1=axR.get_legend_handles_labels(); h2,l2=axr.get_legend_handles_labels()
axR.legend(h1+h2,l1+l2,loc='upper left',bbox_to_anchor=(0.0,1.0),fontsize=9.6,framealpha=0.95)

# ===== 底部: master eq + û + three forces + four phases =====
axk.set_xlim(0,1); axk.set_ylim(0,1)
axk.text(0.5,0.92,'Three-force master equation:   $\\Delta(\\Sigma W^2) = -2\\eta\\,\\Sigma(W\\cdot\\hat u)\\ +\\ \\eta^2\\Sigma\\hat u^2\\ -\\ 2\\eta\\lambda_{wd}\\Sigma W^2$',ha='center',fontsize=13,color='#1a1a1a')
axk.text(0.5,0.79,'Update direction  $\\hat{u}=\\hat{m}/\\sqrt{\\hat{v}}$    (the actual per-step AdamW parameter update)',ha='center',fontsize=10.5,color='#555')
axk.text(0.5,0.69,'$\\hat{m}$ = bias-corrected 1st moment (momentum), $\\hat{m}=m/(1-\\beta_1^{t})$;    $\\hat{v}$ = bias-corrected 2nd moment, $\\hat{v}=v/(1-\\beta_2^{t})$',ha='center',fontsize=9.8,color='#888')
axk.text(0.015,0.50,'Three forces (a):',ha='left',fontsize=10.5,color='#333')
axk.text(0.165,0.50,'$\\blacksquare$ Alignment (learning) $-2\\eta\\Sigma(W\\hat u)$: dominant, 88-94%',ha='left',fontsize=10.3,color=RED)
axk.text(0.625,0.50,'$\\blacksquare$ Injection (noise) $+\\eta^2\\Sigma\\hat u^2$: small',ha='left',fontsize=10.3,color=BLUE)
axk.text(0.875,0.50,'$\\blacksquare$ Decay $-2\\eta\\lambda_{wd}\\Sigma W^2$',ha='left',fontsize=10.3,color=GREEN)
axk.text(0.015,0.27,'Four phases (b):',ha='left',fontsize=10.5,color='#333')
phs=[('1  Plateau: all forces small, $\\lambda$ at init','#5d6d7e'),
     ('2  Rise/overshoot: alignment dominates, $\\lambda$ grows fast','#b9770e'),
     ('3  Overshoot: target falls, $\\lambda$ lags; alignment weakens, decay catches up','#a93226'),
     ('4  Relaxation: decay dominates, $\\eta\\to 0$, learning fades','#1e8449')]
axk.text(0.165,0.27,phs[0][0],ha='left',fontsize=10.3,color=phs[0][1])
axk.text(0.585,0.27,phs[2][0],ha='left',fontsize=10.3,color=phs[2][1])
axk.text(0.165,0.13,phs[1][0],ha='left',fontsize=10.3,color=phs[1][1])
axk.text(0.585,0.13,phs[3][0],ha='left',fontsize=10.3,color=phs[3][1])

fig.subplots_adjust(left=0.045,right=0.93,top=0.95,bottom=0.04,hspace=0.30,wspace=0.14)
fig.savefig('F_HERO_three_force_EN.png',dpi=145,bbox_inches='tight'); print('saved F_HERO_three_force_EN.png')
print(f'data: {len(_st)} real Pythia-70m steps, λ range {obs.min():.1f}-{obs.max():.1f} (×0.001)')
