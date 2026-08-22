#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import json, glob, os, numpy as np, re
BASE="<SET_PATH>"
OUT="<SET_PATH>"
HELD_N=128000; BLK=512
print("[1] importing torch...", flush=True)
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
torch.set_grad_enabled(False)
dev="cuda" if torch.cuda.is_available() else "cpu"
print(f"    torch ok, device={dev}", flush=True)

# Pythia-70m-deduped config
cfg=GPTNeoXConfig(vocab_size=50304,hidden_size=512,num_hidden_layers=6,num_attention_heads=8,
    intermediate_size=2048,rotary_pct=0.25,rotary_emb_base=10000,max_position_embeddings=2048,
    use_parallel_residual=True,layer_norm_eps=1e-5,hidden_act="gelu")

#
clean=np.load(f"{BASE}/token_data/tokens_corrupt/corrupt_rho000_s0.npy")
held=clean[15000000:15000000+HELD_N].astype(np.int64)
nblk=len(held)//BLK
X=torch.tensor(held[:nblk*BLK].reshape(nblk,BLK),dtype=torch.long)
print(f"[2] held-out: {nblk} blocks x {BLK} = {nblk*BLK} clean wikitext tokens", flush=True)

FAM=[("shuffle","_tmp_suite_full/corrupt_grad","tokens_corrupt"),
     ("repeat","_tmp_suite_full/repeat_grad","tokens_repeat"),
     ("replace","_tmp_suite_full/replace_grad","tokens_replace")]
def Dcond(tok,n=2400000):
    t=tok[:n].astype(np.int64); pv=t[:-1]; nx=t[1:]
    def H(x): _,c=np.unique(x,return_counts=True); p=c/c.sum(); return -float(np.sum(p*np.log2(p)))
    return float(H(pv*50304+nx)-H(pv))

def held_loss(sd):
    m=GPTNeoXForCausalLM(cfg);
    miss,unexp=m.load_state_dict(sd,strict=False)
    m.to(dev).eval()
    tot,ntok=0.0,0
    for i in range(nblk):
        ids=X[i:i+1].to(dev)
        out=m(ids,labels=ids)
        tot+=float(out.loss)*(BLK-1); ntok+=(BLK-1)
    del m
    return tot/ntok, len(miss), len(unexp)

rows=[]; outpath=f"{OUT}/p4_heldout_data_v2.json"
i=0
for fam,rdir,tdir in FAM:
    for d in sorted(glob.glob(f"{BASE}/tmp_intermediate/{rdir}/*/")):
        name=d.rstrip("/").split("/")[-1]
        ck=d+"W_ckpts/step8000.pt"; lt=d+"lambda_trajectory_continue.json"; st=d+"v1b_spline_true.jsonl"
        tk=f"{BASE}/token_data/{tdir}/{name}.npy"
        if not all(os.path.exists(x) for x in (ck,lt,st,tk)): continue
        lam=json.load(open(lt)); los=[json.loads(l) for l in open(st)]
        l0,lf=lam[0]["lambda"],lam[-1]["lambda"]; tloss=los[-1]["loss"]
        Dv=Dcond(np.load(tk)); dl=(lf**2-l0**2)*1e4
        sd=torch.load(ck,map_location="cpu",weights_only=False)
        hl,nm,nu=held_loss(sd)
        r=dict(fam=fam,name=name,D=Dv,dlam2=dl,lamf=lf,
               train_loss=tloss,held_loss=hl,gen_gap=hl-tloss,miss=nm,unexp=nu)
        rows.append(r); i+=1
        print(f"  [{i}] {name}: held={hl:.3f} train={tloss:.3f} gap={hl-tloss:+.3f} D={Dv:.2f} dlam2={dl:.2f} (miss{nm}/unexp{nu})", flush=True)
        json.dump(rows,open(outpath,"w"),ensure_ascii=False,indent=1)

#
D=np.array([r["D"] for r in rows]); DL=np.array([r["dlam2"] for r in rows])
HL=np.array([r["held_loss"] for r in rows]); TL=np.array([r["train_loss"] for r in rows])
GG=np.array([r["gen_gap"] for r in rows])
def pc(x,y): return float(np.corrcoef(x,y)[0,1])
def partial(x,y,z):
    Z=np.vstack([z,np.ones_like(z)]).T
    rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    return float(np.corrcoef(rx,ry)[0,1])
print("\n===== held-out results =====",flush=True)
print(f"corr(Δλ², train_loss)         = {pc(DL,TL):+.3f}",flush=True)
print(f"corr(Δλ², held_loss)          = {pc(DL,HL):+.3f}  <- tracks generalization?",flush=True)
print(f"★ pcorr(Δλ², held_loss | D)   = {partial(DL,HL,D):+.3f}",flush=True)
print(f"★ pcorr(Δλ², train_loss | D)  = {partial(DL,TL,D):+.3f}",flush=True)
print(f"corr(Δλ², gen_gap=held-train) = {pc(DL,GG):+.3f}  <- does lambda identify memorization?",flush=True)
print(f"corr(D, held_loss)            = {pc(D,HL):+.3f}",flush=True)
for fam,_,_ in FAM:
    ix=[k for k,r in enumerate(rows) if r["fam"]==fam]
    if len(ix)<3: continue
    print(f"  [{fam}] Δλ²-held {pc(DL[ix],HL[ix]):+.2f} | Δλ²-gap {pc(DL[ix],GG[ix]):+.2f} | mean gap {GG[ix].mean():+.2f}",flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CM={"shuffle":"#2c5aa0","repeat":"#c0392b","replace":"#e6800d"}
fig,ax=plt.subplots(1,3,figsize=(13.5,4.2))
for fam,_,_ in FAM:
    ix=[k for k,r in enumerate(rows) if r["fam"]==fam]
    if not ix: continue
    ax[0].scatter(TL[ix],HL[ix],c=CM[fam],label=fam,s=55,edgecolors="white")
    ax[1].scatter(DL[ix],HL[ix],c=CM[fam],label=fam,s=55,edgecolors="white")
    ax[2].scatter(DL[ix],GG[ix],c=CM[fam],label=fam,s=55,edgecolors="white")
mn=min(TL.min(),HL.min()); mx=max(TL.max(),HL.max())
ax[0].plot([mn,mx],[mn,mx],"--",c="#999",lw=.8); ax[0].set_xlabel("train loss"); ax[0].set_ylabel("held-out clean loss")
ax[0].set_title("(1) train vs held-out\n(above diag = generalization gap)")
ax[1].set_xlabel("dlam2 (weight growth)"); ax[1].set_ylabel("held-out clean loss")
ax[1].set_title(f"(2) dlam2 -> held-out  r={pc(DL,HL):+.2f}\npartial|D r={partial(DL,HL,D):+.2f}")
ax[2].axhline(0,c="#999",lw=.7); ax[2].set_xlabel("dlam2 (weight growth)"); ax[2].set_ylabel("gen gap = held - train")
ax[2].set_title(f"(3) dlam2 -> gen gap  r={pc(DL,GG):+.2f}\n(does lambda flag memorization?)")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); fig.savefig(f"{OUT}/p4_probe_heldout_v2.png",dpi=160,bbox_inches="tight")
print("\nWROTE",f"{OUT}/p4_probe_heldout_v2.png",flush=True)
print("DONE",flush=True)
