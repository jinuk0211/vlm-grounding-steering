"""RBD layer / hyperparameter search for CHAIR recall (user 11-05 18:2x; no rule-based V_obj changes). Controller, CPU only: writes
queues/rbd6.txt for the rbd6_pool.sh workers, scores each stage with the unmodified CHAIR scorer, advances. Decoder: vdd4_decode.py, V_obj unchanged.
Successive halving on the CHAIR test images (user's protocol: selection on test rows, disclosed; rows 200-499 are never used to select):
  stage 1  all candidates, rows 0-60.   Keep 3: CHAIR_s <= current RBD row + 3.4 (2 captions), highest recall (ties F1); if fewer than 3
           qualify, fill with the lowest CHAIR_s.
  stage 2  those 3, rows up to 200.     Winner: CHAIR_s <= RBD + 1.5 (3 captions) and recall > RBD, highest recall. None -> keep the current row.
  stage 3  winner, rows up to 500.      Reported on all 500 and on the unselected rows 200-499, vs vdd Base and vs the RBD row, with and without class orange.
Rule fixed before any candidate result of this round. Log: logs/rbd6.log; RBD6_<od>_DONE per model, RBD6_DONE.
usage: rbd6_ctl.py run | test"""
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';X=O/'runs';Q=O/'queues/rbd6.txt';C=O/'claims_rbd6';TMP=X/'rbd6_tmp';TMP.mkdir(exist_ok=True);C.mkdir(exist_ok=True)
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_rbd6');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];IM={r['id']:r['metadata']['image_id'] for r in refs}
def A(layer,alpha,lam=1,pen=0.5,clip=None):return f'--alpha {alpha} --layer {layer} --lam {lam} --pen {pen}'+(f' --clip {clip}' if clip else '')
MODELS={
 'qwen3':dict(name='Qwen3-VL-8B',model='qwen3',attn='sdpa',od='rbd6_qwen3',src='qchair_test500_all',rbd='s20a1bl1p05',cand={
   's22a1':A(22,1),'s22a2':A(22,2),'s18a1':A(18,1),'s24a1':A(24,1),'s20a05':A(20,0.5),'s20a1l05':A(20,1,0.5),'s20a1l05p025':A(20,1,0.5,0.25),'s20a1c2':A(20,1,clip=2),'s20a1p0':A(20,1,pen=0)}),
 'gemma':dict(name='Gemma 4 E4B',model='gemma',attn='eager',od='rbd6_gemma',src='gchair_test500_all',rbd='s24a1bl1p05',cand={
   's20a2bl1p05':A(20,2),'s16a2':A(16,2),'s18a2':A(18,2),'s22a2':A(22,2),'s20a1':A(20,1),'s20a3':A(20,3),'s20a2l05p025':A(20,2,0.5,0.25),'s20a2c2':A(20,2,clip=2)}),
 'qwen':dict(name='Qwen2.5-VL-7B',model='qwen',attn='sdpa',od='rbd6_qwen25',src='chair_test500_all',rbd='s16a2bl1p05',cand={
   's12a2':A(12,2),'s14a2':A(14,2),'s18a2':A(18,2),'s20a2':A(20,2),'s16a3':A(16,3),'s16a4':A(16,4),'s18a3':A(18,3),'s16a2l05p025':A(16,2,0.5,0.25),'s16a2c2':A(16,2,clip=2),'s16a2p0':A(16,2,pen=0)})}
# 11-05 19:4x user: only Qwen3-VL needs a higher score, Gemma and Qwen2.5 dropped before any of their units ran. Four Qwen3 candidates added after
# the interim look at the first nine (rbd6_peek.py; disclose): deeper steering gave recall, no penalty gave lower CHAIR_s, so their combinations.
MODELS={'qwen3':MODELS['qwen3']};MODELS['qwen3']['cand'].update({'s21a1':A(21,1),'s21a1p0':A(21,1,pen=0),'s22a1p0':A(22,1,pen=0),'s22a1l15':A(22,1,1.5)})
STAGES=[[(0,30),(30,60)],[(60,130),(130,200)],[(200,300),(300,400),(400,500)]]
def log(*a):print(*a,flush=True)
def rows(m,t):
    S=X/MODELS[m]['od'];by={}
    for f in S.glob(f'{t}-*-*.jsonl'):
        for l in open(f):
            if l.endswith('\n') and l.strip():r=json.loads(l);by[r['id']]=r
    return by
def fixed(m,t):
    d=MODELS[m];return {r['id']:r for r in map(json.loads,open(X/d['src']/f"{'base' if t=='base' else d['rbd']}-0-500.jsonl"))}
def get(m,t):return fixed(m,t) if t in('base','rbd') else rows(m,t)
def unit_ok(m,t,a,b):
    by=rows(m,t);return all(i in by for i in ids[a:b])
def failed(m,t,a,b):
    f=C/f"fail-{MODELS[m]['od']}-{t}-{a}-{b}";return f.exists() and len(f.read_text().splitlines())>=2
def score(by,sel):
    f=TMP/'c.json';f.write_text(json.dumps([{'image_id':IM[i],'caption':by[i]['prediction']} for i in sel]))
    d=sc.compute_chair(str(f),'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall'];s={x['image_id']:x for x in d['sentences']}
    hw=[{n for _,n in s[IM[i]]['mscoco_hallucinated_words']} for i in sel]
    per=dict(hal=np.array([bool(x) for x in hw],float),halx=np.array([bool(x-{'orange'}) for x in hw],float),
             gt=np.array([len(set(s[IM[i]]['mscoco_gt_words'])) for i in sel],float),hit=np.array([len(set(s[IM[i]]['mscoco_gt_words'])&set(s[IM[i]]['mscoco_generated_words'])) for i in sel],float))
    t=[by[i]['generated_tokens'] for i in sel]
    return dict(CHAIR_s=100*m['CHAIRs'],noorange=100*per['halx'].mean(),CHAIR_i=100*m['CHAIRi'],recall=100*Rc,F1=100*2*P*Rc/(P+Rc),tok=float(np.mean(t)),at512=int(sum(x>=512 for x in t))),per
def ci(a,b):
    n=len(a['hal']);g=np.random.default_rng(0);ds=[];dx=[];dr=[]
    for _ in range(2000):
        k=g.integers(0,n,n);ds.append(100*(a['hal'][k].mean()-b['hal'][k].mean()));dx.append(100*(a['halx'][k].mean()-b['halx'][k].mean()));dr.append(100*(a['hit'][k].sum()/a['gt'][k].sum()-b['hit'][k].sum()/b['gt'][k].sum()))
    q=lambda v:[float(np.quantile(v,x)) for x in (.025,.975)];return dict(CHAIR_s=q(ds),noorange=q(dx),recall=q(dr))
def fm(c):return f"[{c[0]:+.1f},{c[1]:+.1f}]"
def table(m,tags,sel,title):
    log(f"--- {MODELS[m]['name']} {title} n={len(sel)}");R_={}
    for t in ['base','rbd']+tags:R_[t]=score(get(m,t),sel)
    for t,(v,p) in R_.items():
        s=f"  {t:16s} CHAIR_s {v['CHAIR_s']:5.1f} no-orange {v['noorange']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} F1 {v['F1']:5.1f} tok {v['tok']:4.0f} @512 {v['at512']:3d}"
        if t!='base':x=ci(p,R_['base'][1]);s+=f" | vs Base s{fm(x['CHAIR_s'])} r{fm(x['recall'])}"
        if t not in('base','rbd'):x=ci(p,R_['rbd'][1]);s+=f" | vs RBD s{fm(x['CHAIR_s'])} no-orange{fm(x['noorange'])} r{fm(x['recall'])}"
        log(s)
    return {t:v for t,(v,_) in R_.items()}
def pick3(v,tags):
    r=v['rbd'];ok=sorted([t for t in tags if v[t]['CHAIR_s']<=r['CHAIR_s']+3.4],key=lambda t:(-v[t]['recall'],-v[t]['F1']))
    rest=sorted([t for t in tags if t not in ok],key=lambda t:v[t]['CHAIR_s']);return (ok+rest)[:3]
def pick1(v,tags):
    r=v['rbd'];ok=[t for t in tags if v[t]['CHAIR_s']<=r['CHAIR_s']+1.5 and v[t]['recall']>r['recall']]
    return max(ok,key=lambda t:v[t]['recall']) if ok else None
def prep():
    if 'gemma' not in MODELS:return
    S=X/'rbd6_gemma';S.mkdir(exist_ok=True)
    for a,b in STAGES[0]:
        f=S/f's20a2bl1p05-{a}-{b}.jsonl'
        if not f.exists():shutil.copy(X/f'gchair_test500/s20a2bl1p05-{a}-{b}.jsonl',f)
def main():
    prep();state={m:dict(stage=0,tags=list(d['cand'])) for m,d in MODELS.items()};active=list(MODELS)
    def wq():
        L=['# MODEL ATTN SET OUTDIR TAG START END args']
        for s in range(3):
            for m in active:
                st=state[m];d=MODELS[m]
                if st['stage']!=s:continue
                for a,b in STAGES[s]:
                    for t in st['tags']:
                        if not unit_ok(m,t,a,b):L.append(f"{d['model']} {d['attn']} chair_test500 {d['od']} {t} {a} {b} {d['cand'][t]}")
        p=Path(str(Q)+'.tmp');p.write_text('\n'.join(L)+'\n');p.replace(Q)
    log(f"=== {time.strftime('%T')} rbd6 start");wq()
    while active:
        for m in list(active):
            st=state[m];d=MODELS[m];rng=STAGES[st['stage']]
            for t in list(st['tags']):
                if any(failed(m,t,a,b) for a,b in rng):st['tags'].remove(t);log(f"=== {time.strftime('%T')} {m} {t} dropped (unit failed twice)")
            if not st['tags']:log(f"=== {time.strftime('%T')} RBD6_{d['od']}_DONE (no candidates left)");active.remove(m);continue
            if not all(unit_ok(m,t,a,b) for t in st['tags'] for a,b in rng):continue
            if st['stage']==0:
                v=table(m,st['tags'],ids[:60],'stage 1, rows 0-60');st['tags']=pick3(v,st['tags']);st['stage']=1;log(f"=== {time.strftime('%T')} {m} stage 2 candidates {st['tags']}")
            elif st['stage']==1:
                v=table(m,st['tags'],ids[:200],'stage 2, rows 0-200');w=pick1(v,st['tags'])
                if w is None:log(f"=== {time.strftime('%T')} {m}: no candidate beats the current row. RBD6_{d['od']}_DONE");active.remove(m);continue
                st['tags']=[w];st['stage']=2;log(f"=== {time.strftime('%T')} {m} winner {w} ({d['cand'][w]}); rows 200-500")
            else:
                w=st['tags'][0];table(m,[w],ids,'FINAL all 500');table(m,[w],ids[200:],'FINAL unselected rows 200-499')
                by=rows(m,w);F=X/f"{d['od']}_final";F.mkdir(exist_ok=True);(F/f'{w}-0-500.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
                log(f"=== {time.strftime('%T')} RBD6_{d['od']}_DONE");active.remove(m)
        wq();time.sleep(60) if active else None
    Path(str(Q)+'.done').write_text('done\n');log(f"=== {time.strftime('%T')} RBD6_DONE")
def test():
    prep();m='gemma';tags=['s20a2bl1p05'];v=table(m,tags,ids[:60],'TEST rows 0-60');log('pick3',pick3(v,tags),'pick1',pick1(v,tags));log('s20a2bl1p05 complete on rows 0-60:',all(unit_ok(m,tags[0],a,b) for a,b in STAGES[0]))
    for mm,d in MODELS.items():
        log(mm,'base rows',len(get(mm,'base')),'rbd rows',len(get(mm,'rbd')),'candidates',len(d['cand']))
        assert len(get(mm,'base'))==500 and len(get(mm,'rbd'))==500
    log('TEST_OK')
if __name__=='__main__':(main if sys.argv[1]=='run' else test)()
