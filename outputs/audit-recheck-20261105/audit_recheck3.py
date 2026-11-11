"""Third CPU re-check: lens top-100 by distance to writing x position with CIs; E1 and step 0 without loop captions."""
import glob,json,sys,warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts')
from audit_recheck import auroc,strat,ci,BASE,LL,R
OUT=R/'audit-recheck-20261105';BAND={'qwen':range(22,28),'gemma':range(30,42)}
DB=[1,5,17,65,10**6];DL=['1-4','5-16','17-64','65+'];PBk=[0,64,128,10**6];PL=['0-63','64-127','128+']
for m in ('qwen','gemma'):
    out={'model':m};D=torch.load(LL/f'{m}-topk-0-500.pt',weights_only=False);nodes=D['nodes'];ni={n:i for i,n in enumerate(nodes)};N=len(nodes);band=list(BAND[m])
    bad={json.loads(l)['id'] for l in open(BASE[m]) if 'addCriterion' in json.loads(l)['prediction']}
    # absent baseline per object and position bin
    absH=np.zeros((3,N));absP=np.zeros((3,N));S={}
    for r in D['records']:
        ca=np.array(r['caption_ids']);prev=np.r_[-1,ca[:-1]];h=r['hits'];h=h[np.isin(h[:,0],band)&(h[:,2]<100)];h=h[(h[:,4]!=ca[h[:,1]])&(h[:,4]!=prev[h[:,1]])]
        s=set(map(tuple,h[:,[1,3]].tolist()));S[r['id']]=s;L=r['labels'];GT,M=set(L['gt']),set(L['mentioned']);T=r['T'];bb=np.digitize(np.arange(T),PBk)-1
        for n,i in ni.items():
            if n in GT or n in M:continue
            absP[:,i]+=np.bincount(bb,minlength=3)
        for t,i in s:
            n=nodes[i]
            if n not in GT and n not in M:absH[np.digitize(t,PBk)-1,i]+=1
    base=absH/np.maximum(absP,1);obs=[]  # one row per (object, caption, distance bin, position bin)
    for r in D['records']:
        H=set(r['labels']['hallucinated']);seen={}
        for x in r['mentions']:
            if x['first'] and x['node'] in ni:seen[x['node']]=(x['t'],x['node'] in H)
        for n,(t0,hl) in seen.items():
            i=ni[n];c=np.zeros((4,3));e=np.zeros((4,3));p=np.zeros((4,3))
            for t in range(t0):
                d=np.digitize(t0-t,DB)-1;b=np.digitize(t,PBk)-1;y=(t,i) in S[r['id']];c[d,b]+=y;p[d,b]+=1;e[d,b]+=y-base[b,i]
            obs.append((r['image_id'],hl,r['id'] in bad,c,e,p))
    g=np.array([o[0] for o in obs]);hl=np.array([o[1] for o in obs]);C=np.stack([o[3] for o in obs]);E=np.stack([o[4] for o in obs]);P=np.stack([o[5] for o in obs]);B=np.array([o[2] for o in obs])
    rate=lambda ii,mk,d,b,A=C:A[ii][mk[ii]][:,d,b].sum()/max(P[ii][mk[ii]][:,d,b].sum(),1)
    tb={}
    for nm,mk in (('hallucinated',hl),('correct',~hl)):
        tb[nm]={DL[d]:{PL[b]:{'pct':100*rate(np.arange(len(g)),mk,d,b),'excess':100*rate(np.arange(len(g)),mk,d,b,E),'positions':int(P[mk][:,d,b].sum())} for b in range(3)} for d in range(4)}
    out['top100_by_distance_position']=tb
    allb=np.arange(len(g));cmp={}
    for d in range(4):
        for b in range(3):
            fn=lambda ii:rate(ii,~hl,d,b)-rate(ii,hl,d,b);cmp[f'{DL[d]}|{PL[b]}']={'correct_minus_hall':100*fn(allb),'ci':[100*x for x in ci(fn,g,n=400)]}
    out['correct_minus_hall']=cmp;lt={}
    for d in range(4):
        fn=lambda ii:rate(ii,hl,d,2)-rate(ii,hl,d,0);lt[DL[d]]={'hall_late_minus_early':100*fn(allb),'ci':[100*x for x in ci(fn,g,n=400)]}
        fn2=lambda ii:rate(ii,~hl,d,2)-rate(ii,~hl,d,0);lt[DL[d]]['correct_late_minus_early']=100*fn2(allb);lt[DL[d]]['correct_ci']=[100*x for x in ci(fn2,g,n=400)]
    out['same_distance_late_minus_early']=lt
    # E1 without loop captions
    res={}
    for f in glob.glob(str(R/f'verify-probe-20261030/{m}-e1-g*.pt')):res.update(torch.load(f,weights_only=False)['results'])
    ids={json.loads(l)['metadata']['image_id']:json.loads(l)['id'] for l in open(BASE[m])};e1={}
    for c in ('probe_away_32','random_32','reread_3_all'):
        for nm,ok in (('all',lambda i:True),('no_loops',lambda i:ids[i] not in bad)):
            rows=[(r['image_id'],x) for r in res[c] for x in r['mentions'] if x['first'] and ok(r['image_id'])]
            h=np.array([x['kind']=='hallucinated' for _,x in rows]);d=np.array([x['logp']-x['logp_base'] for _,x in rows]);t0=np.array([x['top1_base'] for _,x in rows]);t1=np.array([x['top1'] for _,x in rows])
            e1[f'{c}|{nm}']={'n_hall':int(h.sum()),'n_correct':int((~h).sum()),'dlogp':[float(d[h].mean()),float(d[~h].mean())],'top1_lost':[float((~t1[h&t0]).mean()),float((~t1[~h&t0]).mean())]}
    out['e1']=e1
    # step 0: attention amount at first mentions
    recs=[r for f in sorted(glob.glob(str(R/f'query-anchor-step0-20261030/{m}-attn-*-*.pt'))) for r in torch.load(f,weights_only=False)['records']]
    full=torch.load(glob.glob(str(R/f'query-anchor-step0-20261030/{m}-attn-*-*.pt'))[0],weights_only=False)['full_attention_layers'];rows=[]
    for r in recs:
        for j,x in enumerate(r['mentions']):
            if x['first']:rows.append((r['image_id'],x['t'],x['kind']=='hallucinated',float(np.nanmean(r['ment'][full][:,:,0,j].float().numpy()))))
    g2=np.array([a[0] for a in rows]);t2=np.array([a[1] for a in rows]);h2=np.array([a[2] for a in rows]);v2=np.array([a[3] for a in rows]);b2=np.digitize(t2,[0,8,16,32,64,128]);nb=np.array([ids[i] not in bad for i in g2])
    out['image_attention_first_mentions']={'correct_gt_hall_position':strat(v2,~h2,b2),'ci':ci(lambda ii:strat(v2[ii],(~h2)[ii],b2[ii]),g2),'no_loops':strat(v2[nb],(~h2)[nb],b2[nb]),'hall_mean':float(v2[h2].mean()),'correct_mean':float(v2[~h2].mean())}
    (OUT/f'{m}-3.json').write_text(json.dumps(out,indent=1,default=float));print('done',m,flush=True)
