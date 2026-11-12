"""THRONE on all 5,000 COCO val2017 images (user 11-09). Rows 0-999: the paper's runs and their judgments (throne_full1000, gemma_throne1000,
*_cf_test_all); rows 1000-4999: the t4k runs, judged per 1,000-image block (t4k_judge.py). The judgments of the pieces are merged per judge and
read with THRONE's own functions (process_aqa_instance_preds as in prepare_aqa_preds, majority_vote_collapsing). Metrics as in xl_score.py and
vdd4_score.py: ALL = official micro (per image; F over the images with P+R>0, as the official scorer's NaN handling), CLS = class-wise.
Two judges (flan-t5 base, large) at threshold 6 = the paper's main columns; three judges (+ xl) at threshold 9 = THRONE's default = the xl columns.
Paired image bootstrap vs the backbone's Base, 2000 resamples, seed 0.
Ranges: 200-999 (check: the two-judge rows must equal tab:throne_full as it stands), 200-N (the report: rows 0-199 picked the final config.),
0-N and 1000-N, with N = 1000 + 1000 x the number of new blocks judged by every judge for every row (or --blocks K).
usage: BENCH=/home/project-b/LJW/bench t4k_score.py [--blocks K] [--json OUT]     (THRONE venv, CPU)"""
import argparse,contextlib,glob,io,json,os,sys
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--blocks',type=int);ap.add_argument('--json',type=Path);a=ap.parse_args()
BENCH=Path(os.environ['BENCH']);T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
sys.path.insert(0,str(T));import throne_score_aqa as A,pycocotools.coco as coco
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');RUNS=O/'runs'
ROWS={'Qwen2.5-VL':('qwen',[('Base','throne_full1000','base'),('REVIS','throne_full1000','revis'),('RUDDER','throne_full1000','rudder_clone'),
                            ('GrAInS','throne_full1000','grains_clone'),('VTI','throne_full1000','vti_author'),
                            ('TETHER, first config.','throne_full1000','s16a1bl2p1'),('TETHER','qwen_cf_test_all','cf_a3bl2p1')]),
      'Gemma 4 E4B':('gemma',[('Base','gemma_throne1000','base'),('REVIS','gemma_throne1000','revis'),('RUDDER','gemma_throne1000','rudder_clone'),
                              ('GrAInS','gemma_throne1000','grains_clone'),('VTI','gemma_throne1000','vti'),
                              ('TETHER, first config.','gemma_throne1000','s24a1bl2p1'),('TETHER','gemma_cf_test_all','cf_a3bl2p1')])}
SETTINGS=((('base','large'),6,'two judges (flan-t5 base, large), threshold 6: main columns'),
          (('base','large','xl'),9,"THRONE's three judges (base, large, xl), threshold 9: xl columns"))
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
full=read(O/'sets/throne_full.jsonl');assert [r['id'] for r in full[:1000]]==[r['id'] for r in read(O/'sets/throne_full1000.jsonl')]
keys=[r['metadata']['image_id'] for r in full];rid=[r['id'] for r in full]
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
gt={k:set(cid[x['category_id']] for x in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def one(pat):
    fs=glob.glob(pat);assert len(fs)<=1,fs;return fs[0] if fs else None
def jdict(M,run,tag,J,nb):
    f=one(str(RUNS/run/'eval'/tag/'combined'/f'*flan-t5-{J}.json'))
    if f is None:return None
    d=json.load(open(f));assert len(d)==1000,(f,len(d))
    for b in range(nb):
        f=one(str(RUNS/f't4k_{M}'/'eval'/tag/f'b{b}-{J}'/'combined'/f'*flan-t5-{J}.json'))
        if f is None:return None
        e=json.load(open(f));assert len(e)==1000 and not set(e)&set(d);d.update(e)
    return d
def blocks_done():
    nb=0
    for b in range(4):
        if all(one(str(RUNS/f't4k_{M}'/'eval'/tag/f'b{b}-{J}'/'combined'/f'*flan-t5-{J}.json')) for _,(M,rows) in ROWS.items() for _,_,tag in rows for J in ('base','large','xl')):nb=b+1
        else:break
    return nb
NB=a.blocks if a.blocks is not None else blocks_done();N=1000+1000*NB
def tokens(M,run,tag):
    t=[r['generated_tokens'] for r in read(RUNS/run/f'{tag}-0-1000.jsonl')]
    if NB:
        new=read(RUNS/f't4k_{M}'/f'{tag}-0-4000.jsonl') if (RUNS/f't4k_{M}'/f'{tag}-0-4000.jsonl').exists() else None
        if new is None:
            rows={}
            for f in glob.glob(str(RUNS/f't4k_parts_{M}'/f'{tag}-*-*.jsonl')):
                if Path(f).name[:-6].rsplit('-',2)[0]==tag:rows.update({r['id']:r for r in read(f)})
            new=[rows[i] for i in rid[1000:N]]
        t+=[r['generated_tokens'] for r in new[:N-1000]]
    return np.array(t,dtype=float)
def preds(M,run,tag,judges):
    dl=[jdict(M,run,tag,J,NB) for J in judges]
    if any(d is None for d in dl):return None
    by=[{int(k):A.process_aqa_instance_preds(v,num_categories=len(idx),sorted_categories_idx_to_name=idx) for k,v in d.items()} for d in dl]
    return {k:[p[k] for p in by] for k in by[0]}
def mats(pr,thr):
    col,dis=A.majority_vote_collapsing(pr,voting_threshold=thr);P=np.zeros((N,80));C=np.zeros_like(P);G=np.zeros_like(P)
    for i,k in enumerate(keys[:N]):
        dq=set(dis[k])
        for g in gt[k]:
            if g not in dq:G[i,n2i[g]]=1
        for p in col[k]:
            if p not in dq:
                P[i,n2i[p]]=1
                if p in gt[k]:C[i,n2i[p]]=1
    return P,C,G
def f(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):return 2*p*r/(p+r),1.25*p*r/(0.25*p+r)
def micro(W,P,C,G):
    np_,nc,ng=P.sum(1),C.sum(1),G.sum(1);ok=(np_>0)&(ng>0);p=np.where(ok,nc/np.maximum(np_,1),0.);r=np.where(ok,nc/np.maximum(ng,1),0.)
    f1,f5=f(p,r);m=~np.isnan(f5);s=W.sum(1)
    return {'P':(W@p)/s,'R':(W@r)/s,'F1':(W[:,m]@f1[m])/W[:,m].sum(1),'F0.5':(W[:,m]@f5[m])/W[:,m].sum(1)}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_;f1,f5=f(pr,rc);return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F1':np.nanmean(f1,1),'F0.5':np.nanmean(f5,1)}
Q=[('ALL','P'),('ALL','R'),('ALL','F1'),('ALL','F0.5'),('CLS','P'),('CLS','R'),('CLS','F1'),('CLS','F0.5')]
def metr(W,m):
    with np.errstate(invalid='ignore'):x,y=micro(W,*m),cls(W,*m)
    return [100*(x if s=='ALL' else y)[q] for s,q in Q]
ranges=[(200,1000,'200-999 (check: tab:throne_full as it stands)')]
if NB:ranges+=[(200,N,f'200-{N-1} (report)'),(0,N,f'0-{N-1}'),(1000,N,f'1000-{N-1} (new rows only)')]
print(f'new blocks judged for every row: {NB} -> N {N}')
out={'n_new_blocks':NB,'N':N,'results':{}}
cache={}
for judges,thr,label in SETTINGS:
    for bb,(M,rows) in ROWS.items():
        for name,run,tag in rows:
            pr=preds(M,run,tag,judges);cache[(judges,bb,name)]=None if pr is None else mats(pr,thr)
for s0,s1,rl in ranges:
    n=s1-s0;rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(2000)]).astype(float);one_=np.ones((1,n))
    for judges,thr,label in SETTINGS:
        print(f'##### rows {rl}, n {n}; {label}')
        for bb,(M,rows) in ROWS.items():
            print(f'== {bb}');print('   '+' '*22+''.join(f'{s+"_"+q:>13s}' for s,q in Q)+'     Tok')
            ref=cache[(judges,bb,'Base')];B0=metr(W,tuple(x[s0:s1] for x in ref)) if ref is not None else None
            for name,run,tag in rows:
                m=cache[(judges,bb,name)]
                if m is None:print(f'   {name:22s} missing judgments');continue
                m=tuple(x[s0:s1] for x in m);v=[float(x[0]) for x in metr(one_,m)];tk=float(tokens(M,run,tag)[s0:s1].mean())
                print(f'   {name:22s}'+''.join(f'{x:13.1f}' for x in v)+f'{tk:8.0f}')
                rec={'metrics':dict(zip([s+'_'+q for s,q in Q],v)),'tokens':tk}
                if B0 is not None and name!='Base':
                    ci=[(float(np.quantile(x-y,.025)),float(np.quantile(x-y,.975))) for x,y in zip(metr(W,m),B0)]
                    print(f"   {'':22s}"+''.join(f"{'[%+.1f,%+.1f]'%c:>13s}" for c in ci));rec['ci_vs_base']=dict(zip([s+'_'+q for s,q in Q],ci))
                out['results'].setdefault(rl,{}).setdefault(f'{len(judges)}j_thr{thr}',{}).setdefault(bb,{})[name]=rec
        print()
# REVIS on Qwen2.5 left every caption of the first 1,000 unchanged (tab:throne_full omits it); the same check on all rows decoded so far
for bb,(M,rows) in ROWS.items():
    try:
        b=read(RUNS/rows[0][1]/'base-0-1000.jsonl');r=read(RUNS/rows[1][1]/'revis-0-1000.jsonl')
        def newrows(tag):
            d={}
            for fp in glob.glob(str(RUNS/f't4k_parts_{M}'/f'{tag}-*-*.jsonl')):
                if Path(fp).name[:-6].rsplit('-',2)[0]==tag:d.update({x['id']:x['prediction'] for x in read(fp)})
            return d
        nbse,nrev=newrows('base'),newrows('revis');common=[i for i in nbse if i in nrev]
        print(f'{bb}: REVIS caption == Base caption on {sum(x["prediction"]==y["prediction"] for x,y in zip(b,r))}/1000 old rows, '
              f'{sum(nbse[i]==nrev[i] for i in common)}/{len(common)} new rows decoded by both')
    except Exception as e:print(bb,'REVIS check failed',e)
if a.json:a.json.write_text(json.dumps(out,indent=1))
