"""Set/queue/join/pick steps of the AMBER vdd3 run (vdd3_chain.sh; plan: amber_v3_runbook.md of session ccf836b1).
  sets       amber_v3dev.jsonl (amber_full1004 rows 100-299 = AMBER ids 101-300), amber_v3test.jsonl (the other 804),
             queues/amber_v3_dev.txt (g1-g6 x 4 chunks of 50)
  join_dev   runs/amber_v3_dev_all/<tag>-0-200.jsonl: g1-g6 from the chunks, base (= vdd Base) and mainMed from
             runs/amber_full1004_ours_all rows 100-299
  pick       reads runs/amber_v3_dev_all/score.json; rule fixed before any run: among g1-g6 and mainMed with
             Cover >= vdd-Base Cover - 1.0 and CHAIR <= vdd-Base CHAIR, the lowest Hal; candidates within 0.5 Hal of it ->
             the highest Cover. Writes runs/amber_v3_pick.txt ("none" if nothing qualifies) and, unless the pick is
             mainMed (rows exist), queues/amber_v3_test.txt (17 chunks of 50 over rows 0-100 and 300-1004)
  join_test  runs/amber_v3_test_vs_vdd (base = vdd Base, mainMed, pick) and runs/amber_v3_test_vs_bench (base = bench Base,
             vddbase, mainMed, pick, revis, rudder_clone, grains_clone, vtiws_v0.9_t0.9), files <tag>-0-804.jsonl
Every piece is checked id for id against the set; nothing existing is re-run."""
import json,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
BASE='--alpha 1 --layer 16 --norm median'
G={'g1':'--gpen 2 --tau 1.0','g2':'--gpen 4 --tau 1.0','g3':'--gpen 2 --tau 0.5','g4':'--gpen 2 --tau 1.0 --stuff exempt',
   'g5':'--gpen 2 --tau 1.0 --hedge 2','g6':'--gpen 2 --tau 1.0 --stuff exempt --hedge 2'}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def write(p,rows):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows))
full=read(O/'sets/amber_full1004.jsonl');aid=[r['metadata']['amber_id'] for r in full];assert aid==sorted(aid) and len(full)==1004
DEV=list(range(100,300));TEST=list(range(0,100))+list(range(300,1004))
assert [aid[i] for i in DEV]==list(range(101,301)) and len(TEST)==804
OURS=O/'runs/amber_full1004_ours_all';BL=O/'runs/amber_full1004_all'
def rows_of(path,idx):
    r=read(path);assert [x['id'] for x in r]==[x['id'] for x in full],path;return [r[i] for i in idx]
def chunks(tag,outdir,spans,ids):
    rows=[]
    for s,e in spans:
        f=O/'runs'/outdir/f'{tag}-{s}-{e}.jsonl';r=read(f) if f.exists() else []
        assert len(r)==e-s,f'{f} incomplete ({len(r)}/{e-s})'
        assert [x['id'] for x in r]==[full[i]['id'] for i in range(s,e)],f
        rows+=r
    assert [x['id'] for x in rows]==ids;return rows
cmd=sys.argv[1]
if cmd=='sets':
    write(O/'sets/amber_v3dev.jsonl',[full[i] for i in DEV]);write(O/'sets/amber_v3test.jsonl',[full[i] for i in TEST])
    q=['# vdd3 dev: AMBER ids 101-300 = rows 100-300 of amber_full1004']
    for s in range(100,300,50):
        for t,a in G.items():q.append(f'amber_full1004 amber_v3_dev {t} {s} {s+50} {BASE} {a}')
    Path(O/'queues/amber_v3_dev.txt').write_text('\n'.join(q)+'\n');print('dev',len(DEV),'test',len(TEST),'units',len(q)-1)
elif cmd=='join_dev':
    d=O/'runs/amber_v3_dev_all';d.mkdir(exist_ok=True);ids=[full[i]['id'] for i in DEV]
    write(d/'base-0-200.jsonl',rows_of(OURS/'base-0-1004.jsonl',DEV));write(d/'mainMed-0-200.jsonl',rows_of(OURS/'mainMed-0-1004.jsonl',DEV))
    for t in G:write(d/f'{t}-0-200.jsonl',chunks(t,'amber_v3_dev',[(s,s+50) for s in range(100,300,50)],ids))
    print('joined dev:',['base','mainMed',*G])
elif cmd=='pick':
    res=json.loads((O/'runs/amber_v3_dev_all/score.json').read_text())['results'];B=res['base']
    qual={t:res[t] for t in (*G,'mainMed') if res[t]['Cover']>=B['Cover']-1.0 and res[t]['CHAIR']<=B['CHAIR']}
    print('vdd Base dev: CHAIR %.1f Cover %.1f Hal %.1f; qualify (Cover >= %.1f, CHAIR <= %.1f): %s'%(B['CHAIR'],B['Cover'],B['Hal'],B['Cover']-1,B['CHAIR'],list(qual)))
    if not qual:(O/'runs/amber_v3_pick.txt').write_text('none\n');print('picked none');sys.exit(0)
    # guard (user 11-04 16:4x, before any dev score): no test unless the pick lowers dev Hal below vdd Base
    better={t:v for t,v in qual.items() if v['Hal']<B['Hal']}
    if not better:
        (O/'runs/amber_v3_pick.txt').write_text('none\n')
        print('picked none: no qualifying candidate has dev Hal below vdd Base %.1f (best %s %.1f); test not run'%(B['Hal'],min(qual,key=lambda t:qual[t]['Hal']),min(v['Hal'] for v in qual.values())));sys.exit(0)
    qual=better
    mh=min(v['Hal'] for v in qual.values());near=[t for t,v in qual.items() if v['Hal']<=mh+0.5];pick=max(near,key=lambda t:qual[t]['Cover'])
    (O/'runs/amber_v3_pick.txt').write_text(pick+'\n')
    nh=[t for t in qual if t not in ('g5','g6')]
    if nh:
        mh2=min(qual[t]['Hal'] for t in nh);print('best non-hedge candidate:',max([t for t in nh if qual[t]['Hal']<=mh2+0.5],key=lambda t:qual[t]['Cover']))
    spans=[(0,50),(50,100)]+[(s,min(s+50,1004)) for s in range(300,1004,50)]
    if pick!='mainMed':
        Path(O/'queues/amber_v3_test.txt').write_text('# vdd3 test: the pick on the 804 images outside dev\n'+''.join(f'amber_full1004 amber_v3_test {pick} {s} {e} {BASE} {G[pick]}\n' for s,e in spans))
    print('picked',pick,'(Hal %.1f Cover %.1f CHAIR %.1f)'%(qual[pick]['Hal'],qual[pick]['Cover'],qual[pick]['CHAIR']))
elif cmd=='join_test':
    pick=(O/'runs/amber_v3_pick.txt').read_text().split()[0];ids=[full[i]['id'] for i in TEST];spans=[(0,50),(50,100)]+[(s,min(s+50,1004)) for s in range(300,1004,50)]
    p=rows_of(OURS/'mainMed-0-1004.jsonl',TEST) if pick=='mainMed' else chunks(pick,'amber_v3_test',spans,ids)
    A=O/'runs/amber_v3_test_vs_vdd';Bd=O/'runs/amber_v3_test_vs_bench'
    vb=rows_of(OURS/'base-0-1004.jsonl',TEST);mm=rows_of(OURS/'mainMed-0-1004.jsonl',TEST)
    write(A/'base-0-804.jsonl',vb);write(A/'mainMed-0-804.jsonl',mm)
    if pick!='mainMed':write(A/f'{pick}-0-804.jsonl',p)
    write(Bd/'base-0-804.jsonl',rows_of(BL/'base-0-1004.jsonl',TEST));write(Bd/'vddbase-0-804.jsonl',vb);write(Bd/'mainMed-0-804.jsonl',mm)
    if pick!='mainMed':write(Bd/f'{pick}-0-804.jsonl',p)
    for t in ('revis','rudder_clone','grains_clone','vtiws_v0.9_t0.9'):write(Bd/f'{t}-0-804.jsonl',rows_of(BL/f'{t}-0-1004.jsonl',TEST))
    print('joined test for pick',pick)
