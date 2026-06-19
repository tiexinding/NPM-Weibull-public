"""4-seed 稳健性图. align% 轨迹 4 seed 叠加, 显示88-94% rise主导一致.
"""
import json,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PF=""
def load_budget(f):  # seed1/2: align_pct直接
    r=[json.loads(l) for l in open(f)]; return np.array([x['step'] for x in r]),np.array([x['align_pct'] for x in r])
def load_spline(f):  # seed3/4: 从force算
    r=[json.loads(l) for l in open(f)]; st=[];ap=[]
    for x in r:
        a,i,d=abs(x['true_align_full']),x['inj_full'],x['dec_full']; st.append(x['step']); ap.append(100*a/(a+i+d))
    return np.array(st),np.array(ap)
seeds=[('seed 1',load_budget('derived_data/self_train/v1b_budget_seed1.jsonl')),
       ('seed 2',load_budget('derived_data/self_train/v1b_budget_seed2.jsonl')),
       ('seed 3',load_spline('derived_data/revision/seed_3/v1b_spline_true.jsonl')),
       ('seed 4',load_spline('derived_data/revision/seed_4/v1b_spline_true.jsonl'))]
cols=['#c0392b','#e67e22','#2980b9','#16a085']
fig,ax=plt.subplots(figsize=(7.5,5))
ax.axhspan(88,94,color='#7fcdbb',alpha=0.22,zorder=0,label='rise-phase band 88-94%')
ax.axvspan(50,5000,color='#f9f4d0',alpha=0.5,zorder=0)
for (nm,(st,ap)),c in zip(seeds,cols):
    ax.plot(st,ap,'-o',ms=2.5,c=c,lw=1.3,label=nm,alpha=0.85)
ax.set_xscale('log'); ax.set_xlabel('Training step',fontsize=11.5); ax.set_ylabel('Alignment share (% of net force)',fontsize=11.5)
ax.set_ylim(40,100); ax.tick_params(labelsize=9.5)
ax.text(500,96,'rise phase',fontsize=9,color='#b9770e',ha='center')
ax.text(15000,55,'saturation\n(align$\\approx$decay)',fontsize=8.5,color='#555',ha='center')
ax.legend(fontsize=9.5,loc='lower left',ncol=2)
ax.set_title('Four random seeds: rise-phase alignment dominance is consistent (88.3-93.8%)',fontsize=10.5,pad=8)
plt.tight_layout(); plt.savefig(PF+"F_4seed_robustness.png",dpi=150,bbox_inches="tight"); plt.close()
print("F_4seed_robustness.png ✓")
for nm,(st,ap) in seeds:
    m=st<=5000; print(f"  {nm}: rise align% {ap[m].min():.1f}-{ap[m].max():.1f}")
