"""Second CPU re-check. (1) Gray-image evidence at object tokens: first vs repeated mentions. (2) Grad x input share, first mentions.
(3) COCO co-occurrence (instances_train2014) of hallucinated objects with what is in the image. (4) Image attention at first mentions."""
import glob,json,sys,warnings
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts')
from score_vti_alpha_sweep import chair_scorer
from audit_recheck import auroc,strat,ci,BASE,LL,R,PB
OUT=R/'audit-recheck-20261105';ANN='data/hallucination-suite/annotations/instances_train2014.json'
sc=chair_scorer();nodes=sorted(set(sc.inverse_synonym_dict.values())-{''})
A=json.load(open(ANN));cat={c['id']:sc.inverse_synonym_dict.get(c['name']) for c in A['categories']};print('categories unmapped',[k for k,v in cat.items() if v is None]);objs=defaultdict(set)
for a in A['annotations']:
    n=cat[a['category_id']]
    if n:objs[a['image_id']].add(n)
for i in A['images']:objs[i['id']]
N=len(objs);cnt=Counter();pair=Counter()
for o in objs.values():
    o=sorted(o);cnt.update(o);pair.update((a,c) for i,a in enumerate(o) for c in o[i+1:])
PMI={a:{c:float(np.log((pair[tuple(sorted((a,c)))]+1)*N/((cnt[a]+1)*(cnt[c]+1)))) for c in nodes if c!=a} for a in nodes}
for m in ('qwen','gemma'):
    out={'model':m,'train_images':N};bad={json.loads(l)['id'] for l in open(BASE[m]) if 'addCriterion' in json.loads(l)['prediction']}
    vd={r['id']:r for f in sorted(glob.glob(str(R/f'visual-decay-20261030/{m}-[0-9]*-[0-9]*.pt'))) if 'partial' not in f for r in torch.load(f,weights_only=False)}
    tx=torch.load(LL/f'{m}-text-0-500.pt',weights_only=False)['records'];im=torch.load(LL/f'{m}-image-0-500.pt',weights_only=False);inodes={n:i for i,n in enumerate(im['nodes'])}
    rows=[]
    for r in tx:
        v=vd[r['id']];e=(v['lens_real'][-1].float()-v['lens_gray'][-1].float()).numpy()
        for x in r['mentions']:
            rows.append((r['image_id'],x['t'],x['kind']=='hallucinated',x['first'],float(e[x['t']]),r['id'] in bad))
    G=np.array([a[0] for a in rows]);T=np.array([a[1] for a in rows]);H=np.array([a[2] for a in rows]);F=np.array([a[3] for a in rows]);E=np.array([a[4] for a in rows]);B=np.array([a[5] for a in rows])
    d={}
    for nm,m_ in (('all_mentions',np.ones(len(T),bool)),('first_mentions',F),('repeat_mentions',~F)):
        for tn,tm in (('t>=64',T>=64),('t<64',T<64),('all_t',T>=0)):
            s=m_&tm;fn=lambda ii:E[s][ii][H[s][ii]].mean()-E[s][ii][~H[s][ii]].mean();d[f'{nm}|{tn}']={'n_hall':int((s&H).sum()),'n_correct':int((s&~H).sum()),'e_hall':float(E[s&H].mean()),'e_correct':float(E[s&~H].mean()),'diff':float(E[s&H].mean()-E[s&~H].mean()),'diff_ci':ci(fn,G[s])}
    out['evidence_decomposition']=d
    s=F&~B;out['evidence_first_t>=64_no_loops']={'e_hall':float(E[s&H&(T>=64)].mean()),'e_correct':float(E[s&~H&(T>=64)].mean())}
    # grad x input on the 200 captions
    f=R/f'visual-reliance-20261030/{m}-0-200.pt'
    if f.exists():
        rec=torch.load(f,weights_only=False,map_location='cpu');J=[]
        for r in rec:
            seen=set()
            for t,j in sorted(r['jacobian'].items(),key=lambda x:int(x[0])):
                if j['kind'] in('hallucinated','correct'):J.append((r['image_id'],int(t),j['kind']=='hallucinated',j['node'] not in seen,j['img_share'][0]));seen.add(j['node'])
        g=np.array([a[0] for a in J]);t=np.array([a[1] for a in J]);h=np.array([a[2] for a in J]);fi=np.array([a[3] for a in J]);v=np.array([a[4] for a in J]);q={}
        for nm,mm in (('all',np.ones(len(v),bool)),('first',fi),('repeat',~fi)):
            bb=np.digitize(t[mm],PB);q[nm]={'n_hall':int((mm&h).sum()),'n_correct':int((mm&~h).sum()),'hall':float(v[mm&h].mean()),'correct':float(v[mm&~h].mean()),'auroc_correct_gt_hall_position':strat(v[mm],~h[mm],bb),'ci':ci(lambda ii:strat(v[mm][ii],(~h[mm])[ii],bb[ii]),g[mm])}
        out['grad_x_input']=q
    # co-occurrence: highest PMI with any GT object
    S=[];C=[];ND=[];GG=[];LN=[]
    for r in im['records']:
        L=r['labels'];GT,M,Hh=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
        for n in nodes:
            c='hall' if n in Hh else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else None
            others=[PMI[n][x] for x in GT if x!=n and x in PMI[n]]
            if c is None or not others:continue
            j=inodes.get(n);S.append(max(others));C.append(c);ND.append(n);GG.append(r['image_id']);LN.append(float(r['max'][-1,j]) if j is not None else np.nan)
    S=np.array(S);C=np.array(C);ND=np.array(ND);GG=np.array(GG);LN=np.array(LN);mk=(C=='hall')|(C=='absent')
    w=[auroc(S[mk&(ND==n)],C[mk&(ND==n)]=='hall') for n in np.unique(ND[C=='hall']) if ((C=='hall')&(ND==n)).sum()>=5]
    sp=[]
    for n in nodes:
        a=(C=='absent')&(ND==n)&~np.isnan(LN)
        if a.sum()>50:sp.append(np.corrcoef(np.argsort(np.argsort(S[a])),np.argsort(np.argsort(LN[a])))[0,1])
    hc=(C=='hall')|(C=='correct')
    # does co-occurrence explain the image-token lens gap? hall vs absent at matched PMI
    pb=np.digitize(S,np.quantile(S[mk],[.2,.4,.6,.8]))
    out['cooccurrence']={'hall_vs_absent':auroc(S[mk],C[mk]=='hall'),'ci':ci(lambda ii:auroc(S[mk][ii],(C[mk]=='hall')[ii]),GG[mk],n=300),'within_object':float(np.nanmean(w)),'objects_within':len(w),'correct_vs_hall(pmi higher for correct)':auroc(S[hc],C[hc]=='correct'),
        'median_pmi':{c:float(np.median(S[C==c])) for c in ('hall','correct','missed','absent')},'share_pmi>0.5':{c:float((S[C==c]>.5).mean()) for c in ('hall','correct','missed','absent')},'lens_vs_pmi_within_object_absent':float(np.nanmean(sp)),
        'lens_hall_vs_absent_raw':auroc(LN[mk],C[mk]=='hall'),'lens_hall_vs_absent_pmi_matched':strat(LN[mk],C[mk]=='hall',pb[mk])}
    (OUT/f'{m}-2.json').write_text(json.dumps(out,indent=1,default=float));print(json.dumps(out,default=float));print('done',m,flush=True)
