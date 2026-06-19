"""C2 per-layer λ heatmap 4-size 英文干净版. 无DRAFT标题.
真Pythia 70m/160m/410m/1b: per-block transmission λ(step×layer) → overshoot-in-time + deep-higher."""
import json,glob,re,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE='/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/NPM_v13_complete'
ROOTS=[BASE+'/40_Dynamics/lambda_dense/data/derived',
       BASE+'/30_NPM_weibull/cascade_v2_20260502/cascade_v3_pull/data/derived']
TRANS=('o','ffn_in','ffn_out')
SIZES=[('70m',6),('160m',12),('410m',24),('1b',16)]

def load(sz):
    """return steps(sorted), per-layer λ matrix [layer×step]. per_block 每条=一层(block_idx, lambda_80)."""
    bystep={}
    for root in ROOTS:
        for f in glob.glob(f'{root}/pythia-{sz}-step*_fit_per_block_v3.json'):
            st=int(re.search(r'-step(\d+)_step',f).group(1))
            d=json.load(open(f)); bylayer={}
            for e in d['per_block']:
                L=e.get('block_idx'); lam=e.get('lambda_80')
                if L is not None and lam is not None: bylayer[L]=lam
            if bylayer: bystep[st]=bylayer
    steps=sorted(s for s in bystep if s>=0)
    layers=sorted({L for s in steps for L in bystep[s]})
    M=np.full((len(layers),len(steps)),np.nan)
    for j,s in enumerate(steps):
        for i,L in enumerate(layers):
            if L in bystep[s]: M[i,j]=bystep[s][L]*1e3
    return steps,layers,M

fig,axes=plt.subplots(2,2,figsize=(15,10))
for ax,(sz,nl) in zip(axes.ravel(),SIZES):
    steps,layers,M=load(sz)
    im=ax.imshow(M,aspect='auto',origin='lower',cmap='viridis',interpolation='nearest')
    ax.set_title(f'pythia-{sz} ({len(layers)} layers): per-layer $\\lambda$ ($\\times 10^{{3}}$)',fontsize=12)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([f'{s//1000}k' if s>=1000 else str(s) for s in steps],rotation=60,fontsize=7)
    ax.set_xlabel('Training step',fontsize=10)
    ax.set_ylabel('Layer (0=shallow)',fontsize=10)
    yt=range(0,len(layers),max(1,len(layers)//6)); ax.set_yticks(list(yt)); ax.set_yticklabels([layers[i] for i in yt],fontsize=8)
    plt.colorbar(im,ax=ax,label='$\\lambda\\times 10^{3}$',fraction=0.046,pad=0.04)
plt.tight_layout()
fig.savefig('paper_figures/F_perlayer_heatmap_4size_EN.png',dpi=150,bbox_inches='tight'); print('saved F_perlayer_heatmap_4size_EN.png')
for sz,_ in SIZES:
    steps,layers,M=load(sz); pj=np.nanargmax(np.nanmean(M,axis=0))
    print(f'pythia-{sz}: peak step~{steps[pj]}, λ range {np.nanmin(M):.1f}-{np.nanmax(M):.1f}, deep/shallow {np.nanmax(M[-1])/np.nanmax(M[0]):.2f}x')
