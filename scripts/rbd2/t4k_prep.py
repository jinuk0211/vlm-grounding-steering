"""THRONE on the other 4,000 COCO val2017 images (user 11-09: the paper's THRONE runs cover the first 1,000 of the 5,000; run the rest with every
baseline and the evaluation, GPUs 0 and 5). Rows 1000-4999 of sets/throne_full.jsonl = sets/throne_r4000.jsonl, decoded with the decoders and
arguments of the 1,000-image runs (queues bl_throne_full.txt, full1000_ours.txt, gemma_bl.txt, gemma_ours.txt, v4_*_test.txt):
  Qwen2.5  base, revis, rudder_clone (sdpa), grains_clone, vti_author (eager)  baseline_bench.py
           s16a1bl2p1 (TETHER first config.)                                     vdd_decode.py --alpha 1 --layer 16 --lam 2 --pen 1
           cf_a3bl2p1 (TETHER)                                                   vdd4_decode.py --attn sdpa --alpha 3 --layer 16 --lam 2 --pen 1 --vobj qwen-vobj2cf
  Gemma 4  base, revis, rudder_clone, grains_clone, vti                          gemma_bench.py
           s24a1bl2p1                                                            vdd_decode.py --model gemma --attn eager --alpha 1 --layer 24 --lam 2 --pen 1
           cf_a3bl2p1                                                            vdd4_decode.py --attn eager --alpha 3 --layer 24 --lam 2 --pen 1 --vobj gemma-vobj2cf
Units are block-major (4 blocks of 1,000 rows), so each finished block is complete for every method and can be judged (t4k_judge.py) and scored
(t4k_score.py) before the rest.
usage: t4k_prep.py sets | queues | repro | join | status"""
import json,sys,time,glob,os
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';S=O/'sets';Q=O/'queues';RUNS=O/'runs'
V=str(R/'outputs/rbd-20261030/collect')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
# kind, tag, chunk, args (identical to the 1,000-image queues)
METH={'qwen':[('qv','s16a1bl2p1',50,'--alpha 1 --layer 16 --lam 2 --pen 1'),
              ('qb','vti_author',50,'--attn eager'),
              ('v4q','cf_a3bl2p1',100,f'--attn sdpa --alpha 3 --layer 16 --lam 2 --pen 1 --vobj {V}/qwen-vobj2cf.json'),
              ('qb','grains_clone',200,'--attn eager'),('qb','rudder_clone',200,'--attn sdpa'),('qb','base',200,'--attn sdpa'),('qb','revis',200,'--attn sdpa')],
      'gemma':[('gv','s24a1bl2p1',50,'--alpha 1 --layer 24 --lam 2 --pen 1'),
               ('v4g','cf_a3bl2p1',50,f'--attn eager --alpha 3 --layer 24 --lam 2 --pen 1 --vobj {V}/gemma-vobj2cf.json'),
               ('gb','rudder_clone',100,''),('gb','base',100,''),('gb','revis',100,''),('gb','vti',100,''),('gb','grains_clone',100,'')]}
# rows of the stored 1,000-image runs that were decoded on an A6000 (GPUs 0 and 5 are A6000s; A5000 and A6000 greedy captions differ)
REPRO={'qwen':{'base':(0,2,'throne_full1000'),'revis':(0,2,'throne_full1000'),'rudder_clone':(0,2,'throne_full1000'),'grains_clone':(0,2,'throne_full1000'),
               'vti_author':(100,102,'throne_full1000'),'s16a1bl2p1':(500,502,'throne_full1000'),'cf_a3bl2p1':(250,252,'qwen_cf_test_all')},
       'gemma':{t:(100,102,'gemma_throne1000') for t in ('base','revis','rudder_clone','grains_clone','vti')}|
               {'s24a1bl2p1':(0,2,'gemma_throne1000'),'cf_a3bl2p1':(350,352,'gemma_cf_test_all')}}
BLOCK=1000;NB=4
def parts(M,tag):
    rows={}
    for f in glob.glob(str(RUNS/f't4k_parts_{M}'/f'{tag}-*-*.jsonl')):
        if Path(f).name[:-6].rsplit('-',2)[0]!=tag:continue
        for r in read(f):rows[r['id']]=r
    return rows
cmd=sys.argv[1]
if cmd=='sets':
    full=read(S/'throne_full.jsonl');old=read(S/'throne_full1000.jsonl');assert len(full)==5000
    assert [(r['id'],r['image_sha256']) for r in full[:1000]]==[(r['id'],r['image_sha256']) for r in old]
    for r in full[1000:]:assert Path(r['image']).exists(),r['image']
    (S/'throne_r4000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in full[1000:]));print('throne_r4000',len(full)-1000,full[1000]['id'],'..',full[-1]['id'])
elif cmd=='queues':
    rep=[];dec=['# t4k_prep.py: THRONE rows 1000-4999 (sets/throne_r4000.jsonl), block-major']
    for M in ('qwen','gemma'):
        for k,t,c,a in METH[M]:s,e,_=REPRO[M][t];rep.append(f'{k} throne_full1000 t4k_repro_{M} {t} {s} {e} {a}'.rstrip())
    jud=[]
    for b in range(NB):
        for i in range(max(len(METH['qwen']),len(METH['gemma']))):
            for M in ('gemma','qwen'):
                if i>=len(METH[M]):continue
                k,t,c,a=METH[M][i]
                for s in range(b*BLOCK,(b+1)*BLOCK,c):dec.append(f'{k} throne_r4000 t4k_parts_{M} {t} {s} {s+c} {a}'.rstrip())
        for J in ('base','large','xl'):
            for M in ('qwen','gemma'):
                for k,t,c,a in METH[M]:jud.append(f'{M} {t} {b} {J} '+','.join(f'{s}-{s+c}' for s in range(b*BLOCK,(b+1)*BLOCK,c)))
    (Q/'t4k_repro.txt').write_text('\n'.join(rep)+'\n');(Q/'t4k_decode.txt').write_text('\n'.join(dec)+'\n');(Q/'t4k_judge.txt').write_text('\n'.join(jud)+'\n')
    print('repro units',len(rep),'decode units',len(dec)-1,'judge units',len(jud))
elif cmd=='repro':
    bad=0
    for M in ('qwen','gemma'):
        for t,(s,e,od) in REPRO[M].items():
            ref=read(RUNS/od/f'{t}-0-1000.jsonl')[s:e];f=RUNS/f't4k_repro_{M}'/f'{t}-{s}-{e}.jsonl'
            new={r['id']:r for r in read(f)} if f.exists() else {}
            for r in ref:
                n=new.get(r['id'])
                if n is None:print(f'{M:5s} {t:13s} {r["id"]:13s} MISSING');bad+=1;continue
                a,b_=r['prediction'],n['prediction'];k=next((i for i,(x,y) in enumerate(zip(a,b_)) if x!=y),min(len(a),len(b_)))
                same=a==b_;bad+=not same
                print(f'{M:5s} {t:13s} {r["id"]:13s} {"identical" if same else f"DIFFERS at char {k} of {len(a)}/{len(b_)}"}  gpu {n.get("gpu","-")}  {n["elapsed_seconds"]:.0f}s')
    print('REPRO_OK' if bad==0 else f'REPRO_BAD {bad}');sys.exit(1 if bad else 0)
elif cmd=='join':
    refs=read(S/'throne_r4000.jsonl');ok=True
    for M in ('qwen','gemma'):
        (RUNS/f't4k_{M}').mkdir(exist_ok=True)
        for k,t,c,a in METH[M]:
            rows=parts(M,t);miss=[r['id'] for r in refs if r['id'] not in rows]
            if miss:print(M,t,'missing',len(miss));ok=False;continue
            out=[rows[r['id']] for r in refs];assert all(o['metadata']['image_id']==r['metadata']['image_id'] for o,r in zip(out,refs))
            (RUNS/f't4k_{M}'/f'{t}-0-4000.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in out));print(M,t,'joined 4000')
    sys.exit(0 if ok else 1)
elif cmd=='status':
    refs=read(S/'throne_r4000.jsonl');ids=[r['id'] for r in refs];now=time.time()
    print(f'{"model":6s} {"tag":13s} {"rows":>5s}/4000  {"s/row":>6s}  judged blocks base/large/xl')
    for M in ('qwen','gemma'):
        for k,t,c,a in METH[M]:
            rows=parts(M,t);el=[r['elapsed_seconds'] for r in rows.values()]
            jb=[sum(1 for b in range(NB) if glob.glob(str(RUNS/f't4k_{M}'/'eval'/t/f'b{b}-{J}'/'combined'/f'*flan-t5-{J}.json'))) for J in ('base','large','xl')]
            print(f'{M:6s} {t:13s} {len(rows):5d}/4000  {(sum(el)/len(el) if el else 0):6.1f}  {jb[0]}/{jb[1]}/{jb[2]}')
