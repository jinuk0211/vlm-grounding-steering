"""ALOHa on HAT through the official code (DavidMChan/aloha @7da6b90): ALOHa.__call__, MPNetSimilarity and DETR labels at
threshold 0.9 (add_object_detections), as METRICS['aloha'] with 3 in-context examples. Only additions: the parser reads the
cache of hat_parse.py, embeddings are cached (same numbers), and an object list whose alternatives ('a or b') multiply past
CAP keeps the first alternative of the later groups (counted as capped).
  detect                             DETR labels of every set image -> $HAT_RUN/detections.json
  validate PARSER                    HAT's own 400 captions vs the gold labels: AP of 1 - ALOHa (paper, GPT-3.5 + DETR: 48.62)
  score PARSER RUNDIR START END OUT  every complete run on rows START-END of the set; paired bootstrap vs 'base' (and vs
                                     'vddbase' for vdd rows), 2000 resamples, seed 0. OUT.json = table, OUT.per_image.json = rows.
Columns (ALOHa and meanNN x100, higher is better; shares in %):
  ALOHa    official caption score: mean over images of min over objects of the Hungarian (1-1) similarity. No objects -> nan
           (official -inf), counted in noobj. The official aggregate leaves out objects with no partner, so on captions with
           more objects than references (T>R column) the min runs over the references' partners only.
  meanNN   each object's best reference (no 1-1 matching), mean over the caption's objects, mean over images.
  Flag     objects with best-reference similarity <= 0.5 (the paper's flag level), % of all objects; #Flag per image.
  RefCov   reference objects with some caption object above 0.5, % (a recall column; not in the ALOHa paper).
  same     captions identical to 'base'.
HAT_ALOHA_FIX=1 (user 11-09): the official aggregate with its bug fixed. _compute_alohao_scores means to add a 0 for every caption
object without a Hungarian partner ("Add entries of 0 similarity for every unmatched target word"), but
`similarity_scores + np.array([0]) * n_unmatched` adds 0 elementwise and drops them. Fixed: min (or mean) over all caption objects,
unmatched ones at 0, i.e. over the 'similarity' of the official matches list. Everything else unchanged."""
import json,os,sys
from pathlib import Path
import numpy as np
from hat_common import H,HAT_JSON,SET,key,load_cache,load_runs,read
CAP=64
FLAG=0.5
FIX=os.environ.get('HAT_ALOHA_FIX')=='1'

def build(parser):
    from aloha.metrics.aloha import ALOHa
    from aloha.object_parser.base import ObjectParser
    from aloha.string_similarity import MPNetSimilarity
    cache=load_cache(parser)
    class Cached(ObjectParser):
        def extract_objects_single_caption(self,c):return cache[key('single',c)]
        def extract_objects_multiple_captions(self,cs):return cache[key('multi','\n'.join(cs))]
    class Sim(MPNetSimilarity):
        def __init__(self):super().__init__();self.e={}
        def vec(self,strings):
            new=[s for s in dict.fromkeys(strings) if s not in self.e]
            if new:self.e.update(zip(new,self._model.encode(new,show_progress_bar=False,batch_size=256)))
            return np.stack([self.e[s] for s in strings])
        def __call__(self,t,r):return self.vec(t)@self.vec(r).T
    class Capped(ALOHa):
        capped=0;memo={}
        def _parse_objects_from_llm_response(self,s,reference_mode=False):
            k=(s,reference_mode)
            if k not in self.memo:
                groups=[list(dict.fromkeys(g)) for g in super()._parse_objects_from_llm_response(s,reference_mode)];n=1;out=[]
                for g in groups:
                    if not g:continue
                    if n*len(g)>CAP:g=g[:1];Capped.capped+=1
                    n*=len(g);out.append(g)
                self.memo[k]=out
            return self.memo[k]
        if FIX:
            def _compute_alohao_scores(self,t,r,aggregate_method='min'):
                s,m=super()._compute_alohao_scores(t,r,aggregate_method)
                if not t or not r:return s,m   # official early returns, no aggregate bug there
                v=np.array([x['similarity'] for x in m],float);assert len(v)==len(t)
                return float(v.mean() if aggregate_method=='mean' else v.min()),m
    return Capped(name='aloha',object_parser=Cached,similarity_measure=Sim,num_reference_examples=3,num_target_examples=3,debug=False)

def one(metric,caption,refs,dets):
    """Official score + per-object numbers of one caption."""
    s,info=metric(caption,refs,None,dets);ref_groups=info['reference_objects']
    if not np.isfinite(s):return {'aloha':float('nan'),'mean_nn':float('nan'),'n_t':0,'n_unm':0,'n_flag':0,'n_cov':0,'n_ref':len(ref_groups),'noobj':1}
    m=info['matches'] or [];t=[x['target_word'] for x in m]
    flat=list(dict.fromkeys(a for g in ref_groups for a in g))
    sim=metric._similarity_measure(t,flat) if t and flat else np.zeros((len(t),1))
    best=sim.max(1);at=dict(zip(flat,sim.max(0).tolist()))
    return {'aloha':float(s),'mean_nn':float(best.mean()),'n_t':len(t),'n_unm':sum(x['ref_word'] is None for x in m),
            'n_flag':int((best<=FLAG).sum()),'n_cov':sum(any(at[a]>FLAG for a in g) for g in ref_groups),'n_ref':len(ref_groups),'noobj':0}

def detect():
    import torch
    from aloha.utils import add_object_detections
    torch.set_num_threads(8);rows=read(SET)
    with torch.no_grad():s=add_object_detections([{'image':r['image']} for r in rows],'',image_key='image')
    out={r['id']:x['detections'] for r,x in zip(rows,s)};H.mkdir(parents=True,exist_ok=True);(H/'detections.json').write_text(json.dumps(out))
    print('detections',len(out),'images, mean labels per image %.1f'%np.mean([len(v) for v in out.values()]))

def validate(parser):
    from sklearn.metrics import average_precision_score as ap
    D=json.loads((H/'detections.json').read_text());by={r['metadata']['image_path']:r['id'] for r in read(SET)}
    metric=build(parser);hat=json.loads(HAT_JSON.read_text(encoding='utf-8'));y=[];s=[]
    for x in hat:
        r=one(metric,x['caption'],x['references'],D[by[x['image_path']]]);y.append(x['contains_hallucination'])
        s.append(1-(1.0 if np.isnan(r['aloha']) else r['aloha']))   # no objects, no hallucination
    print(f'HAT reproduction ({parser}): AP {100*ap(y,s):.2f}  (paper GPT-3.5 + DETR 48.62, CHAIRs 36.85; share of hallucinated samples {100*np.mean(y):.2f})  n {len(y)}')

def metrics(x,k=None):
    k=np.arange(len(x['aloha'])) if k is None else k
    return {'ALOHa':100*np.nanmean(x['aloha'][k]),'meanNN':100*np.nanmean(x['mean_nn'][k]),'Flag':100*x['n_flag'][k].sum()/max(x['n_t'][k].sum(),1),
            '#Flag':x['n_flag'][k].mean(),'RefCov':100*x['n_cov'][k].sum()/max(x['n_ref'][k].sum(),1)}

def score(parser,run_dir,a,b,out):
    rows=read(SET)[a:b];ids=[r['id'] for r in rows];runs,partial=load_runs(run_dir,ids);assert 'base' in runs,f'no complete base on rows {a}-{b}'
    D=json.loads((H/'detections.json').read_text());metric=build(parser);per={};res={}
    for tag,x in runs.items():
        p=[one(metric,x[r['id']]['prediction'],r['metadata']['references'],D[r['id']]) for r in rows]
        per[tag]={k:np.array([q[k] for q in p],float) for k in p[0]};r0=x[ids[0]]
        res[tag]={**metrics(per[tag]),'#Obj':per[tag]['n_t'].mean(),'T>R':100*(per[tag]['n_unm']>0).mean(),'noobj':int(per[tag]['noobj'].sum()),
                  'tok':float(np.mean([x[i]['generated_tokens'] for i in ids])),'at512':sum(x[i]['generated_tokens']>=512 for i in ids),
                  'same':sum(x[i]['prediction']==runs['base'][i]['prediction'] for i in ids),'setting':r0.get('setting')}
    rng=np.random.default_rng(0);B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)];Q=('ALOHa','meanNN','Flag','#Flag','RefCov')
    def ci(t,ref):
        d=np.array([[metrics(per[t],k)[q]-metrics(per[ref],k)[q] for q in Q] for k in B])
        return {q:[float(np.nanquantile(d[:,j],.025)),float(np.nanquantile(d[:,j],.975))] for j,q in enumerate(Q)}
    for t in runs:
        if t!='base':res[t]['ci_base']=ci(t,'base')
        if t!='vddbase' and 'vddbase' in runs and runs[t][ids[0]].get('decoder'):res[t]['ci_vddbase']=ci(t,'vddbase')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    Path(f'{out}.json').write_text(json.dumps({'parser':parser,'aloha_fix':FIX,'rows':[a,b],'capped':type(metric).capped,'results':res,'partial':partial},indent=1))
    Path(f'{out}.per_image.json').write_text(json.dumps({'ids':ids,'per':{t:{k:v.tolist() for k,v in p.items()} for t,p in per.items()}}))
    print(f'rows {a}-{b} ({len(ids)} images), parser {parser}, aggregate bug fixed {FIX}, capped object lists {type(metric).capped}'+(f', incomplete runs {partial}' if partial else ''))
    print(f"{'':24s}{'ALOHa':>6s}{'meanNN':>7s}{'Flag':>6s}{'#Flag':>6s}{'RefCov':>7s}{'#Obj':>6s}{'T>R':>6s}{'noobj':>6s}{'tok':>5s}{'@512':>5s}{'same':>5s}")
    for t,v in sorted(res.items(),key=lambda kv:(kv[0]!='base',kv[0]!='vddbase',kv[0])):
        cs=lambda name:' '.join(f"d{q}[{c[0]:+.1f},{c[1]:+.1f}]" for q,c in v[name].items() if q in ('ALOHa','meanNN','Flag','RefCov')) if name in v else ''
        print(f"{t:24s}{v['ALOHa']:6.1f}{v['meanNN']:7.1f}{v['Flag']:6.1f}{v['#Flag']:6.2f}{v['RefCov']:7.1f}{v['#Obj']:6.1f}{v['T>R']:6.1f}{v['noobj']:6d}{v['tok']:5.0f}{v['at512']:5d}{v['same']:5d}  {cs('ci_base')}"+(f"  vs vddbase {cs('ci_vddbase')}" if 'ci_vddbase' in v else ''))

if __name__=='__main__':
    c=sys.argv[1]
    if c=='detect':detect()
    elif c=='validate':validate(sys.argv[2])
    else:score(sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),sys.argv[6])
