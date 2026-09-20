import json,sys,numpy as np,torch
sys.argv=['x']; exec(open('eval_slice2.py').read().split('if __name__')[0].replace('X = torch.tensor','X = None #'))
R='<selfavg_mechanism>/08_paper5_draft/data/'
d1=json.load(open(R+'p5v3_gain_edit_eval_v1.json'))
out=[]
for c in d1['conditions']:
    p,m=c['path'],c['mode']
    sd2=edit(SD,p,m,a=c.get('a',1.0),rng=np.random.default_rng(c.get('seed',0)),decile=c.get('decile'))
    num=den=0.0; lf=[]
    for l in range(L):
        k1,k2,ax1,ax2,h1,h2=fields(SD,l,p)
        for k in (k1,k2):
            num+=float((sd2[k]-SD[k]).pow(2).sum()); den+=float(SD[k].pow(2).sum())
        # log-factor rms on the gain (both sides same except balance)
        g=h1+h2
        if m=='flatten': lf.append(float((c['a']*g/2).pow(2).mean().sqrt()))
        elif m=='shuffle': pi=torch.tensor(np.random.default_rng(c['seed']).permutation(len(g))); lf.append(float(((g[pi]-g)/2).pow(2).mean().sqrt()))
        elif m=='decile':
            q=torch.quantile(g,torch.tensor([c['decile']/10,(c['decile']+1)/10])); sel=(g>=q[0])&(g<=q[1]) if c['decile']==9 else (g>=q[0])&(g<q[1])
            lf.append(float(torch.where(sel,g/2,torch.zeros_like(g)).pow(2).mean().sqrt()))
        else: lf.append(float(((h1-h2)/2).pow(2).mean().sqrt()))
    rel=num/den; out.append({**c,'relF2':rel,'logf_rms':float(np.mean(lf))})
    print(f"{p} {m:8s} {c.get('a',c.get('seed',c.get('decile','')))!s:5s} dloss {c['dloss']:+.4f}  relF2 {rel:.4f}  logf_rms {np.mean(lf):.4f}  dloss/relF2 {c['dloss']/rel if rel>1e-9 else 0:.2f}")
json.dump(out,open('gain_edit_perturbation_v1.json','w'),indent=1)
