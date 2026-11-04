"""Summarize visual_reliance.py outputs.

H1  image reliance falls as the caption gets longer (all caption tokens / control tokens by position bin)
H2  hallucinated object tokens rely less on the image than correct object tokens (overall and within position bins)
H3  the same object relies more on the image when asked as a question than when written in the caption
"""
import glob,json,sys
from pathlib import Path
import numpy as np
import torch

def auroc(pos,neg):
    pos=np.asarray(pos);neg=np.asarray(neg)
    if len(pos)==0 or len(neg)==0:return float('nan')
    allv=np.concatenate([pos,neg]);r=allv.argsort().argsort()+1.0
    # average ranks for ties
    vals,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True);csum=np.cumsum(cnt);avg=csum-(cnt-1)/2;r=avg[inv]
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def boot_diff(a,b,n=2000,seed=0):
    a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(seed)
    d=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(n)]
    return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]

BINS=[0,16,32,64,96,128,192,256,384,513]
def binof(t):
    for i in range(len(BINS)-1):
        if BINS[i]<=t<BINS[i+1]:return f'{BINS[i]}-{BINS[i+1]-1}'
    return 'tail'

def main(path,model):
    recs=[r for f in sorted(glob.glob(str(path/f'{model}-*.pt'))) for r in torch.load(f,map_location='cpu',weights_only=False)]
    out={'model':model,'captions':len(recs)}
    # H1: all caption tokens
    by={}
    for r in recs:
        att=r['attn_img'].float();vcd=(r['logp_real'].float()-r['logp_gray'].float())
        L=att.shape[0];late=att[L//2:].mean(0);early=att[:L//2].mean(0)
        for t in range(len(vcd)):
            b=by.setdefault(binof(t),{'attn_early':[],'attn_late':[],'vcd':[],'jac':[]})
            b['attn_early'].append(float(early[t]));b['attn_late'].append(float(late[t]));b['vcd'].append(float(vcd[t]))
        for t,j in r['jacobian'].items():
            if j['kind']=='control':by.setdefault(binof(int(t)),{'attn_early':[],'attn_late':[],'vcd':[],'jac':[]})['jac'].append(j['img_share'][0])
    out['H1_by_position']={k:{m:(float(np.mean(v)) if v else None) for m,v in d.items()}|{'n_tokens':len(d['vcd']),'n_jac':len(d['jac'])}
                           for k,d in sorted(by.items(),key=lambda x:int(x[0].split('-')[0]) if x[0]!='tail' else 9999)}
    # H2: object tokens
    obj={'hallucinated':[],'correct':[]}
    for r in recs:
        att=r['attn_img'].float();L=att.shape[0];late=att[L//2:].mean(0);vcd=(r['logp_real'].float()-r['logp_gray'].float())
        for t,j in r['jacobian'].items():
            t=int(t)
            if j['kind'] in obj and t<len(vcd):obj[j['kind']].append({'t':t,'jac':j['img_share'][0],'vcd':float(vcd[t]),'attn_late':float(late[t]),'logp':j['logp']})
    h2={}
    for m in ('jac','vcd','attn_late','logp'):
        hv=[x[m] for x in obj['hallucinated']];cv=[x[m] for x in obj['correct']]
        h2[m]={'hallucinated_mean':float(np.mean(hv)),'correct_mean':float(np.mean(cv)),'diff_CI95':boot_diff(hv,cv),'auroc_correct_gt_hallucinated':auroc(cv,hv)}
        # within position bins (weighted mean of per-bin differences)
        diffs=[];w=[]
        for b in set(binof(x['t']) for x in obj['hallucinated']):
            hb=[x[m] for x in obj['hallucinated'] if binof(x['t'])==b];cb=[x[m] for x in obj['correct'] if binof(x['t'])==b]
            if len(hb)>=5 and len(cb)>=5:diffs.append(np.mean(hb)-np.mean(cb));w.append(len(hb))
        h2[m]['position_matched_diff']=float(np.average(diffs,weights=w)) if diffs else None
    out['H2_object_tokens']={'n':{k:len(v) for k,v in obj.items()},**h2}
    # H3: question vs caption for the same (caption, object)
    pairs={'hallucinated':[],'correct':[]}
    for r in recs:
        first={}
        for t,j in sorted(r['jacobian'].items(),key=lambda x:int(x[0])):
            if j['kind'] in pairs and j['node'] not in first:first[j['node']]=j
        for nd,q in r['question'].items():
            if nd in first:pairs[q['kind']].append((first[nd]['img_share'][0],q['img_share'][0],q['margin']))
    h3={}
    for k,v in pairs.items():
        if not v:continue
        c=np.array([x[0] for x in v]);q=np.array([x[1] for x in v]);mg=np.array([x[2] for x in v])
        h3[k]={'n':len(v),'caption_img_share':float(c.mean()),'question_img_share':float(q.mean()),'question_minus_caption_CI95':boot_diff(q,c),
               'frac_question_higher':float((q>c).mean()),'denied_when_asked':float((mg<0).mean())}
    out['H3_question_vs_caption']=h3
    print(json.dumps(out,indent=1));(path/f'{model}-reliance-summary.json').write_text(json.dumps(out,indent=1))

if __name__=='__main__':
    main(Path(sys.argv[1]),sys.argv[2])
