"""B-sel图. Selection(Q/K) vs Transmission align% 轨迹.
🔴caveat: Selection非Weibull-1.20(k0.28-0.51), 只力趋势不挂λ. """
import json,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PF="paper_figures/"
rows=[json.loads(l) for l in open(PF+'cloud_revision/bsel/v1b_spline_true.jsonl')]
st=np.array([r['step'] for r in rows])
def pct(al,inj,dec): a=np.abs(al); return 100*a/(a+inj+dec)
ta=np.array([pct(r['true_align_full'],r['inj_full'],r['dec_full']) for r in rows])
sa=np.array([pct(r['sel_align'],r['sel_inj'],r['sel_dec']) for r in rows])
fig,ax=plt.subplots(figsize=(7.5,5))
ax.axvspan(50,5000,color='#f9f4d0',alpha=0.5,zorder=0)
ax.plot(st,ta,'-o',ms=3,c='#d95f0e',lw=1.6,label='Transmission (O+FFN, $k\\approx$1.20)')
ax.plot(st,sa,'-s',ms=3,c='#8e44ad',lw=1.6,label='Selection (Q/K, $k$=0.28-0.51, non-Weibull)')
ax.set_xscale('log'); ax.set_xlabel('Training step',fontsize=11.5); ax.set_ylabel('Alignment share (% of net force)',fontsize=11.5)
ax.set_ylim(30,100); ax.tick_params(labelsize=9.5)
ax.text(500,96,'rise phase',fontsize=9,color='#b9770e',ha='center')
# 标注移左下角空白区, 不压线
ax.legend(fontsize=9.5,loc='lower left')
ax.set_title('Selection (Q/K) vs Transmission force budget: same rise dominance, different saturation balance',fontsize=9.5,pad=8)
plt.tight_layout(); plt.savefig(PF+"F_bsel_select_vs_trans.png",dpi=150,bbox_inches="tight"); plt.close()
print("F_bsel_select_vs_trans.png ✓")
