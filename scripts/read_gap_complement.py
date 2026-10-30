"""Does the vision-side probe add information beyond the model's own Yes/No margin?
Refits the probe config chosen in analysis-<model>.json (train images only), then on POPE:
2-fold-by-image logistic on [margin, logit(probe)] vs margin alone (AUROC, accuracy),
and how many FN items the probe sees as strongly as a typical TP."""
import json,math,random,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
import read_gap_analyze_v3 as A
E0=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/read-gap-e0-20261030');PS=Path('/home/project-b/LJW/vlm-grounding-steering/data/coco-train2014-probe5000')
out={}
for model in sys.argv[1:]:
    an=json.loads((E0/f'analysis-{model}.json').read_text())[model];tag=an['pope_tag'];ch=an['vision_probe']['chosen']
    rows=A.read(E0/f'{tag}-rows.jsonl');names=json.loads((PS/'categories.json').read_text())
    # force the chosen config: restrict selection to it by monkeypatching feats loop is unnecessary; refit directly
    tr_states=torch.load(E0/f'{model}-train-vision-states.pt');items={r['id']:r for r in A.read(PS/'list.jsonl')}
    Y=torch.zeros(len(tr_states['ids']),80)
    for i,k in enumerate(tr_states['ids']):
        for c in items[k]['objects']:Y[i,c]=1
    X=A.feats(tr_states['pooled'],ch['layer'],ch['pool']);Xs,mu,sd=A.standardize(X);W,b=A.fit_logreg(Xs,Y)
    ps=torch.load(E0/f'{tag}-vision-states.pt');gi={g:i for i,g in enumerate(ps['groups'])}
    Z=((A.feats(ps['pooled'],ch['layer'],ch['pool'])-mu)/sd)@W+b;idx={n:i for i,n in enumerate(names)}
    import re
    pz=[float(Z[gi[r['group_id']],idx[re.match(r'Is there an? (.+) in the image\?',r['prompt']).group(1).strip().lower()]]) for r in rows]
    m=[r['margin'] for r in rows];gold=[r['label']=='yes' for r in rows];groups=[r['group_id'] for r in rows]
    res={'probe_auroc':A.auroc(pz,gold),'margin_auroc':A.auroc(m,gold)}
    accs={'margin':[],'combined':[]};aucs={'margin':[],'combined':[]}
    for s in range(20):
        keys=sorted(set(groups));random.Random(s).shuffle(keys);half=set(keys[:len(keys)//2])
        sc={'margin':[0]*len(rows),'combined':[0]*len(rows)};pred={'margin':[0]*len(rows),'combined':[0]*len(rows)}
        for fold in (half,set(keys)-half):
            tr=[i for i,g in enumerate(groups) if g not in fold];te=[i for i,g in enumerate(groups) if g in fold]
            for name,cols in (('margin',[m]),('combined',[m,pz])):
                F=torch.tensor([[c[i] for c in cols] for i in range(len(rows))],dtype=torch.float32)
                Ft,fm,fs=A.standardize(F[tr]);w,bb=A.fit_logreg(Ft,torch.tensor([[float(gold[i])] for i in tr]),wd=0,steps=500)
                z=(((F-fm)/fs)@w+bb)[:,0]
                # threshold tuned on the training fold for accuracy
                ztr=[float(z[i]) for i in tr];t=A.best_threshold(ztr,[gold[i] for i in tr])
                for i in te:sc[name][i]=float(z[i]);pred[name][i]=float(z[i])>t
        for name in sc:aucs[name].append(A.auroc(sc[name],gold));accs[name].append(A.metrics(pred[name],gold)['acc'])
    res['cv20']={k:{'auroc':sum(aucs[k])/20,'acc':sum(accs[k])/20,'acc_range':[min(accs[k]),max(accs[k])]} for k in aucs}
    base=[x>0 for x in m];tp=sorted(pz[i] for i in range(len(rows)) if gold[i] and base[i])
    q25=tp[len(tp)//4];q50=tp[len(tp)//2];fn=[pz[i] for i in range(len(rows)) if gold[i] and not base[i]]
    res['FN_seen_as_strong_as_TP']={'n_FN':len(fn),'above_TP_q25':sum(x>=q25 for x in fn),'above_TP_median':sum(x>=q50 for x in fn)}
    out[model]=res
print(json.dumps(out,indent=1));(E0/'analysis-complement.json').write_text(json.dumps(out,indent=1))
