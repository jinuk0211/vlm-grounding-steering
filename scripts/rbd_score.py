"""Score caption rows on the same subset of CHAIR test images with the unmodified CHAIR scorer (test ground truth).

Existing baseline rows are reused as they are (no regeneration); every method is restricted to the image ids of
the first --n rows of the test set, so all methods are compared on identical images. Paired bootstrap over
images for CHAIR_s and recall differences against the first method given (Base).
usage: rbd_score.py --model qwen --n 100 --out DIR name=path.jsonl [name=path.jsonl ...]
"""
import argparse,json
from pathlib import Path
import numpy as np
import read_gap_collect_v2 as C
from score_vti_alpha_sweep import chair_scorer

def main(a):
    a.out.mkdir(parents=True,exist_ok=True);sc=chair_scorer()
    test=C.read(C.MODELS[a.model]/'original-vti-gpu1-final/chair/base.jsonl' if a.model=='qwen' else C.MODELS[a.model]/'chair/base.jsonl')
    ids=[r['metadata']['image_id'] for r in test[:a.n]];res={};per={}
    for spec in a.runs:
        name,path=spec.split('=',1);rows={r['metadata']['image_id']:r for r in C.read(path)}
        missing=[i for i in ids if i not in rows]
        if missing:print(name,'missing',len(missing),'images; skipped');continue
        cap=a.out/f'{name}.captions.json';cap.write_text(json.dumps([{'image_id':i,'caption':rows[i]['prediction']} for i in ids]))
        d=sc.compute_chair(str(cap),'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];R=m['Recall']
        s={x['image_id']:x for x in d['sentences']}
        per[name]={'hal':np.array([len(s[i]['mscoco_hallucinated_words'])>0 for i in ids],float),
                   'n_gen':np.array([len(s[i]['mscoco_generated_words']) for i in ids],float),
                   'n_hal':np.array([len(s[i]['mscoco_hallucinated_words']) for i in ids],float),
                   'n_gt':np.array([len(set(s[i]['mscoco_gt_words'])) for i in ids],float),
                   'n_hit':np.array([len(set(s[i]['mscoco_gt_words'])&set(s[i]['mscoco_generated_words'])) for i in ids],float)}
        res[name]={'CHAIR_s':100*m['CHAIRs'],'CHAIR_i':100*m['CHAIRi'],'recall':100*R,'object_F1':100*2*P*R/(P+R),
                   'mean_tokens':float(np.mean([rows[i].get('generated_tokens',0) for i in ids]))}
    base=next(iter(per));rng=np.random.default_rng(0);B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)]
    for name in per:
        if name==base:continue
        x,b=per[name],per[base]
        ds=[100*(x['hal'][k].mean()-b['hal'][k].mean()) for k in B]
        dr=[100*(x['n_hit'][k].sum()/x['n_gt'][k].sum()-b['n_hit'][k].sum()/b['n_gt'][k].sum()) for k in B]
        res[name]['dCHAIR_s_ci95']=[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))]
        res[name]['drecall_ci95']=[float(np.quantile(dr,.025)),float(np.quantile(dr,.975))]
    (a.out/'score.json').write_text(json.dumps({'model':a.model,'n_images':len(ids),'results':res},indent=1))
    print(f"{a.model}, first {len(ids)} CHAIR test images")
    for k,v in res.items():
        ci=f"  dCHs [{v['dCHAIR_s_ci95'][0]:+.1f},{v['dCHAIR_s_ci95'][1]:+.1f}] drec [{v['drecall_ci95'][0]:+.1f},{v['drecall_ci95'][1]:+.1f}]" if 'dCHAIR_s_ci95' in v else ''
        print(f"{k:16s} CHs {v['CHAIR_s']:5.1f} CHi {v['CHAIR_i']:5.2f} rec {v['recall']:5.1f} F1 {v['object_F1']:5.2f} tok {v['mean_tokens']:4.0f}{ci}")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--n',type=int,default=100)
    p.add_argument('--out',type=Path,required=True);p.add_argument('runs',nargs='+');main(p.parse_args())
