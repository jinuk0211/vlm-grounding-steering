"""CHAIR dev (the 100 COCO train2014 images of data/chair-dev-train2014-100) for the right half of tab:hallucination_all_models, in place of AMBER
(user 11-05 20:3x; the user writes up how this split was used). Base, RUDDER, GrAInS, VTI and RBD on Qwen2.5, Qwen3 and Gemma. RBD at the table's setting of
the backbone, no selection here. REVIS is out until its decision, as on HAT. Both Bases, as in the table: baselines against the bench Base, RBD and VTI against the
vdd Base. Existing rows are reused where the setting is the same: rows 0-59 of Qwen2.5 vdd Base and VTI, of Gemma vdd Base, and rows 0-29 of Gemma RBD.
  prep            sets/chair_dev100.jsonl, reused rows, queues/chair_dev100.txt (hat_pool.sh lines, runs/chair_dev100_MODEL)
  score MODEL     runs/chair_dev100_MODEL_all (base = bench Base) and _vdd (base = vdd Base) through score_sets.py, then the table cells in logs/cdev_table_MODEL.tex
  scoretest       the scorer path on the stored chair_dev60 rows: must give their known numbers (Base 38.3 / 7.6 / 66.8, main 36.7 / 7.5 / 68.4)"""
import json,os,shutil,sys
from pathlib import Path
L=Path(os.environ.get('HAT_L','/home/project-b/LJW'));R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102';RUN='chair_dev100';N=100
SET=O/f'sets/{RUN}.jsonl';SPANS=[(0,60),(60,N)]
METHODS={'qwen':[('bench','base','--attn sdpa'),('vdd','vddbase','--attn sdpa'),('vdd','s16a2bl1p05','--attn sdpa --layer 16 --alpha 2 --lam 1 --pen 0.5'),
                 ('bench','rudder_clone','--attn sdpa'),('bench','grains_clone','--attn eager'),('vti','vtiws_v0.9_t0.9','')],
         'qwen3':[('bench','base','--attn sdpa'),('vdd','vddbase','--attn sdpa'),('vdd','s20a1bl1p05','--attn sdpa --layer 20 --alpha 1 --lam 1 --pen 0.5'),
                  ('bench','rudder_clone','--attn sdpa'),('grains','grains_clone','--attn eager'),('bench','vti','--attn sdpa')],
         'gemma':[('bench','base',''),('vdd','vddbase','--attn eager'),('vdd','s24a1bl1p05','--attn eager --layer 24 --alpha 1 --lam 1 --pen 0.5'),
                  ('bench','rudder_clone',''),('bench','grains_clone',''),('bench','vti','')]}
# tag: (sources in row order, setting the rows must have; None for a run that records no setting)
REUSE={'qwen':{'vddbase':(['chair_dev60/base-0-60'],dict(alpha=0.0,lam=0.0,pen=0.0,attn='sdpa')),'vtiws_v0.9_t0.9':(['chair_dev60_vtiws/vtiws_v0.9_t0.9-0-60'],None)},
       'gemma':{'vddbase':(['gchair_dev60/base-0-30','gchair_dev60/base-30-60'],dict(alpha=0.0,lam=0.0,pen=0.0,attn='eager')),
                's24a1bl1p05':(['gchair_dev60/s24a1bl1p05-0-30'],dict(alpha=1.0,layer=24,lam=1.0,pen=0.5,attn='eager'))}}
def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def rundir(m):return O/'runs'/f'{RUN}_{m}'

def prep():
    rows=read(R/'data/chair-dev-train2014-100/chair.jsonl');assert len(rows)==N and rows[:60]==read(O/'sets/chair_dev60.jsonl')
    SET.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');ids=[r['id'] for r in rows]
    for m,tags in REUSE.items():
        rundir(m).mkdir(parents=True,exist_ok=True)
        for t,(src,want) in tags.items():
            dst=rundir(m)/f'{t}-0-60.jsonl';got=[x for s in src for x in read(O/'runs'/f'{s}.jsonl')]
            assert [x['id'] for x in got]==ids[:len(got)],(m,t,'ids')
            for x in got:assert want is None or all((x.get('setting') or {}).get(k)==v for k,v in want.items()),(m,t,x.get('setting'))
            if not dst.exists():dst.write_text(''.join(json.dumps({**x,'method':t})+'\n' for x in got),encoding='utf-8')
            print('reused',m,t,len(got),'rows')
    q=[f'{m} {k} {t} {s} {e} {a}'.rstrip() for m,ms in METHODS.items() for s,e in SPANS for k,t,a in ms]
    p=O/f'queues/{RUN}.txt';p.write_text('\n'.join(q)+'\n',newline='\n');print(p.name,len(q),'units')

def run_scorer(d,name,n):
    os.environ.setdefault('RBD2_SETS',str(O/'sets'));os.environ.setdefault('RBD2_CHAIR_TEST_STATE',str(R/'data/hallucination-suite/chair-scorer-state.json'))
    sys.path.insert(0,str(R/'scripts/rbd2'));import score_sets as S
    S.STATE[RUN]=S.STATE['chair_dev60'];S.SET=name;S.N=n;S.main(d)

def score(m):
    ids=[r['id'] for r in read(SET)];rows={}
    for k,t,a in METHODS[m]:
        by={}
        for s,e in SPANS:
            f=rundir(m)/f'{t}-{s}-{e}.jsonl'
            if f.exists():by.update({x['id']:x for x in read(f)})
        if all(i in by for i in ids):rows[t]=[by[i] for i in ids]
        else:print(m,t,'INCOMPLETE',sum(i in by for i in ids),'rows')
    assert 'base' in rows and 'vddbase' in rows,f'{m}: no complete Bases: {sorted(rows)}'
    A,B=Path(f'{rundir(m)}_all'),Path(f'{rundir(m)}_vdd');A.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
    def w(d,t,r):(d/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r),encoding='utf-8')
    for t,r in rows.items():
        w(A,t,r)
        if t=='vddbase':w(B,'base',r)
        elif t!='base':w(B,t,r)
    print(f'=== {m}: paired bootstrap vs bench Base');run_scorer(A,RUN,N)
    print(f'=== {m}: paired bootstrap vs vdd Base');run_scorer(B,RUN,N)
    a=json.loads((A/'score.json').read_text())['results'];b=json.loads((B/'score.json').read_text())['results']
    names={'base':'Base','rudder_clone':'RUDDER','grains_clone':'GrAInS','vti':'VTI','vtiws_v0.9_t0.9':'VTI'}
    def cell(v,c,q,f):
        x=c.get('ci',{}).get(q);return f'{v[q]:{f}}'+('$^{*}$' if x and (x[0]>0 or x[1]<0) else '')
    out=[]
    for k,t,_ in METHODS[m]:
        if t not in a:continue
        vdd=k=='vdd' or names.get(t)=='VTI';v=a[t];c=b.get(t,{}) if vdd else v
        out.append(f"{'Base, RBD code path' if t=='vddbase' else 'RBD (Ours)' if k=='vdd' else names[t]} & {cell(v,c,'CHAIR_s','.1f')} & {v['CHAIR_i']:.2f} & {cell(v,c,'recall','.1f')} & {v['tokens']:.0f} \\\\")
    (O/f'logs/cdev_table_{m}.tex').write_text('\n'.join(out)+'\n',encoding='utf-8');print('\n'.join(out))

def scoretest():
    d=O/RUN/'scoretest';d.mkdir(parents=True,exist_ok=True)
    for t in ('base','main'):shutil.copy(O/f'runs/chair_dev60/{t}-0-60.jsonl',d)
    run_scorer(d,'chair_dev60',60);x=json.loads((d/'score.json').read_text())['results']
    assert (round(x['base']['CHAIR_s'],1),round(x['base']['CHAIR_i'],1),round(x['base']['recall'],1))==(38.3,7.6,66.8) and round(x['main']['CHAIR_s'],1)==36.7,x
    print('scoretest ok')

if __name__=='__main__':
    c=sys.argv[1]
    if c=='prep':prep()
    elif c=='score':score(sys.argv[2])
    else:scoretest()
