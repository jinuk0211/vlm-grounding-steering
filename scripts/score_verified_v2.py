"""Score self-verified decoding captions with the official CHAIR scorer and compare
with the verified five-method captions on the same 500 images (image-level paired
bootstrap, 10,000 resamples). CPU only."""
import argparse,glob,json,os,re
from pathlib import Path
import numpy as np
from qwen_gpu0_migration_api import save
from score_vti_alpha_sweep import chair_scorer
import read_gap_collect_v2 as C

FINAL={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair','gemma':C.MODELS['gemma']/'chair'}
OTHERS={'qwen':['base','revis','rudder_clone','grains_clone'],'gemma':['base','revis','rudder_clone','grains_clone','vti']}

def stats(detail,order):
    by={str(s['image_id']):s for s in detail['sentences']};out=[]
    for i in order:
        s=by[str(i)];gen=s['mscoco_generated_words'];gt=set(s['mscoco_gt_words'])
        out.append([len(s['mscoco_hallucinated_words'])>0,len(s['mscoco_hallucinated_words']),len(gen),len(set(gen)&gt),len(gt)])
    return np.array(out,dtype=float)

def agg(a):
    return {'CHAIR_s':a[:,0].mean(),'CHAIR_i':a[:,1].sum()/max(a[:,2].sum(),1),'recall':a[:,3].sum()/max(a[:,4].sum(),1)}

def boot(a,b,n=10000,seed=0):
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),(n,len(a)));d={}
    for k,f in (('CHAIR_s',lambda x:x[...,0].mean(-1)),('CHAIR_i',lambda x:x[...,1].sum(-1)/x[...,2].sum(-1)),('recall',lambda x:x[...,3].sum(-1)/x[...,4].sum(-1))):
        v=f(b[idx])-f(a[idx]);d[k]=[float(np.quantile(v,.025)),float(np.quantile(v,.975))]
    return d

def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    refs=C.read(C.MODELS[a.model]/'data/chair.jsonl');order=[r['metadata']['image_id'] for r in refs]
    rows={}
    for f in sorted(glob.glob(str(a.dir/f'{a.tag}-*-*.jsonl'))):
        for r in C.read(f):rows[r['id']]=r
    if a.partial:refs=[r for r in refs if r['id'] in rows];order=[r['metadata']['image_id'] for r in refs]
    else:assert set(rows)=={r['id'] for r in refs},(len(rows),'incomplete')
    rows=[rows[r['id']] for r in refs];out=a.dir/(f'{a.tag}-partial{len(rows)}-score' if a.partial else f'{a.tag}-score');out.mkdir(exist_ok=a.partial)
    scorer=chair_scorer();path=out/'captions.json'
    save(path,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in rows])
    detail=scorer.compute_chair(str(path),'image_id','caption');save(out/'chair.official.json',detail)
    mine=stats(detail,order);res={'model':a.model,'tag':a.tag,'official':detail['overall_metrics'],'methods':{}}
    toks=lambda rs:{'mean_tokens':float(np.mean([r['generated_tokens'] for r in rs])),'at_cap':int(sum(r['generated_tokens']>=512 for r in rs)),
                    'mean_words':float(np.mean([len(re.findall(r'[A-Za-z]{2,}',r['prediction'])) for r in rs]))}
    base=None
    for m in OTHERS[a.model]:
        d=json.loads((FINAL[a.model]/f'{m}.chair.official.json').read_text());s=stats(d,order)
        if a.partial:d['overall_metrics']='n/a (subset)'
        keep={r['id'] for r in refs};mr=[r for r in C.read(FINAL[a.model]/f'{m}.jsonl') if r['id'] in keep]
        res['methods'][m]={**agg(s),**toks(mr),'official':d['overall_metrics']}
        if m=='base':base=s
        else:res['methods'][m]['vs_base_CI95']=boot(base,s)
    res['methods'][a.tag]={**agg(mine),**toks(rows),'vs_base_CI95':boot(base,mine),
        'rejections_total':sum(len(r['rejections']) for r in rows),'captions_with_rejection':sum(bool(r['rejections']) for r in rows),
        'verifications_total':sum(len(r['verified_margins']) for r in rows),'forced_accepts':sum(r['forced_accepts'] for r in rows),
        'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in rows]))}
    save(out/'score.json',res)
    for m,v in res['methods'].items():
        ci=v.get('vs_base_CI95',{})
        print(f"{m:16s} CHs {100*v['CHAIR_s']:5.1f} CHi {100*v['CHAIR_i']:5.2f} rec {100*v['recall']:5.1f} tok {v['mean_tokens']:5.0f} cap {v['at_cap']:3d}",
              {k:[round(100*x,2) for x in c] for k,c in ci.items()})
    print('n',len(rows),'official',res['official'],'rejections',res['methods'][a.tag]['rejections_total'],'captions_with_rej',res['methods'][a.tag]['captions_with_rejection'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dir',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--partial',action='store_true')
    main(p.parse_args())
