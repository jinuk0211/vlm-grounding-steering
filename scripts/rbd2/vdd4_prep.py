"""Queues, joins and the pick of the colour-fixed-V_obj THRONE run (vdd4_chain.sh; MODEL = qwen | gemma).
User 11-04 16:5x: the setting is chosen on the FIRST 200 TEST IMAGES (throne_full1000 rows 0-199 = sets/throne_t200),
not on a separate dev split, and only then run on the other 800; the paper states this. Rows 0-199 of the pick are reused.
  sets                     sets/throne_t200.jsonl = throne_full1000 rows 0-199
  queue_sel MODEL          queues/v4_<model>_sel.txt: the 4 colour-fixed candidates x 4 chunks of 50 on rows 0-199
  join MODEL sel TAG...    runs/<model>_cf_t200_all/<tag>-0-200.jsonl
  join MODEL test TAG      runs/<model>_cf_test_all/<tag>-0-1000.jsonl = rows 0-199 (selection run) + 200-999 (test run)
  pick MODEL SCORE_TXT     rule fixed before any result: a cf_* candidate is eligible only if its F0.5_CLS on the 200 beats
                           BOTH bench Base and the old-V_obj config (the earlier test config); none eligible -> 'none', no
                           test. Among eligible ones: highest F0.5_CLS; within 0.5 of it -> highest F0.5_ALL.
                           Writes runs/<model>_cf_pick.txt and queues/v4_<model>_test.txt (16 chunks of 50, rows 200-999)."""
import json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
cmd=sys.argv[1];model=sys.argv[2] if len(sys.argv)>2 else 'qwen';L=16 if model=='qwen' else 24;ATTN='sdpa' if model=='qwen' else 'eager'
V=f'/home/project-b/LJW/vlm-grounding-steering/outputs/rbd-20261030/collect/{model}-vobj2cf.json'
CAND={'cf_a1bl2p1':f'--attn {ATTN} --alpha 1 --layer {L} --lam 2 --pen 1 --vobj {V}',
      'cf_a2bl2p1':f'--attn {ATTN} --alpha 2 --layer {L} --lam 2 --pen 1 --vobj {V}',
      'cf_a3bl2p1':f'--attn {ATTN} --alpha 3 --layer {L} --lam 2 --pen 1 --vobj {V}',
      'cf_a1bl3p1':f'--attn {ATTN} --alpha 1 --layer {L} --lam 3 --pen 1 --vobj {V}'}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
full=read(O/'sets/throne_full1000.jsonl');ref=[r['id'] for r in full];assert len(ref)==1000
def chunks(outdir,t,spans):
    rows=[]
    for s,e in spans:
        f=O/f'runs/{outdir}/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else []
        assert len(r)==e-s and [x['id'] for x in r]==ref[s:e],f'{f} incomplete or misordered';rows+=r
    return rows
def write(p,rows):p.parent.mkdir(exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
if cmd=='sets':
    (O/'sets/throne_t200.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in full[:200]));print('throne_t200 = throne_full1000 rows 0-199')
    (O/'sets/throne_t800.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in full[200:]));print('throne_t800 = throne_full1000 rows 200-999 (not used for selection)')
elif cmd=='queue_sel':
    q=['# vdd4 selection: throne_full1000 rows 0-199 (first 200 test images)']+[f'throne_full1000 {model}_cf_t200 {t} {s} {s+50} {a}' for s in range(0,200,50) for t,a in CAND.items()]
    (O/f'queues/v4_{model}_sel.txt').write_text('\n'.join(q)+'\n');print(len(q)-1,'units')
elif cmd=='join':
    stage=sys.argv[3]
    for t in sys.argv[4:]:
        if stage=='sel':write(O/f'runs/{model}_cf_t200_all/{t}-0-200.jsonl',chunks(f'{model}_cf_t200',t,[(s,s+50) for s in range(0,200,50)]))
        else:
            rows=chunks(f'{model}_cf_t200',t,[(s,s+50) for s in range(0,200,50)])+chunks(f'{model}_cf_test',t,[(s,s+50) for s in range(200,1000,50)])
            assert len({json.dumps({k:v for k,v in (r.get('setting') or {}).items()},sort_keys=True) for r in rows})==1;write(O/f'runs/{model}_cf_test_all/{t}-0-1000.jsonl',rows)
        print('joined',stage,t)
elif cmd=='pick':
    rx=re.compile(r'^(\S+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|',re.M)
    rows={m[1]:(float(m[9]),float(m[5])) for m in rx.finditer(Path(sys.argv[3]).read_text())}   # tag -> (F0.5_CLS, F0.5_ALL)
    assert {'base','old_a1bl2p1',*CAND}<=set(rows),rows
    bcls=rows['base'][0];ocls=rows['old_a1bl2p1'][0];cand={t:rows[t] for t in CAND}
    elig={t:v for t,v in cand.items() if v[0]>bcls and v[0]>ocls}
    if not elig:
        (O/f'runs/{model}_cf_pick.txt').write_text('none\n')
        print('picked none: best cf F0.5_CLS %.1f does not beat bench Base %.1f and old V_obj %.1f on the 200; test not continued'%(max(v[0] for v in cand.values()),bcls,ocls));sys.exit(0)
    top=max(v[0] for v in elig.values());near=[t for t,v in elig.items() if v[0]>=top-0.5];pick=max(near,key=lambda t:elig[t][1])
    (O/f'runs/{model}_cf_pick.txt').write_text(pick+'\n')
    (O/f'queues/v4_{model}_test.txt').write_text('# vdd4 test: the pick on throne_full1000 rows 200-999\n'+''.join(f'throne_full1000 {model}_cf_test {pick} {s} {s+50} {CAND[pick]}\n' for s in range(200,1000,50)))
    print('picked',pick,'F0.5_CLS %.1f F0.5_ALL %.1f (bench Base %.1f, old %.1f); eligible %s; within 0.5 of the best: %s'%(rows[pick][0],rows[pick][1],bcls,ocls,list(elig),near))
