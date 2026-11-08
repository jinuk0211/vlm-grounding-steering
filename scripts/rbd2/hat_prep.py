"""Set, queues, pick and progress of the HAT + ALOHa run (hat_chain.sh; models qwen, gemma).
  set                 sets/$HAT_RUN.jsonl: the 385 unique images of HAT's 400 samples in file order (the public file has no val
                      split), prompt and max_new_tokens of chair_test150 unless HAT_PROMPT / HAT_MAX_TOK are set, COCO's references
                      from the HAT file. Images: existing COCO val2014 copies, others from images.cocodataset.org.
  queue_dev           queues/$HAT_RUN_dev.txt: selection on rows 0-59: bench Base, vdd Base, 7 RBD candidates of each model.
  pick MODEL JSON     rule fixed before any result (score JSON of rows 0-59): a candidate is eligible only if its ALOHa beats BOTH
                      bench Base and vdd Base and its RefCov >= vdd Base RefCov - 2. Among eligible ones: highest ALOHa; within 0.5
                      of it -> highest RefCov. None eligible -> 'none': the model's RBD does not continue, baselines do.
                      Writes $HAT_RUN/pick-MODEL.txt.
  queue_test MODELS   queues/$HAT_RUN_test.txt: the picks + both Bases on rows 60-384, the other baselines on all rows.
  queue_bg MODELS     queues/$HAT_RUN_bg.txt, lowest priority for joined workers: units of a later model that need no pick (Bases, baselines, candidates on the
                      selection rows), so that no worker idles while the chain parses and scores the model before it. The chain later lists the same units and skips them.
  status              rows done per model and method, and the remaining GPU-hours from the seconds per image so far.
  pilot MODELS        HAT_RUN=pilot_chair500: no decoding. The CHAIR 500 captions that exist (vdd Base and the paper's RBD row) with
                      COCO's references, to read ALOHa's direction before the HAT decode.
Baselines as on AMBER 1004 / THRONE 1000: Qwen baseline_bench.py (base, rudder_clone sdpa; grains_clone eager) and
vti_ws_run.py at 0.9/0.9; Gemma gemma_bench.py (all eager, VTI = own implementation).
REVIS is out of these queues (user 11-05 20:2x): baselines.py is a paper-based own implementation, not antgroup/Revis. A REVIS run placed later in
runs/${HAT_RUN}_MODEL as revis-0-385.jsonl (row format as the others) is scored like any other method by hat_aloha.py score.
Queue lines: MODEL KIND TAG START END args... -> runs/${HAT_RUN}_MODEL/TAG-START-END.jsonl."""
import collections,concurrent.futures as cf,hashlib,json,os,shutil,sys,urllib.request
from pathlib import Path
from hat_common import H,HAT_JSON,O,R,RUN,SET,read,rundir
N=385;SEL=60
SPANS=[(0,SEL)]+[(s,min(s+50,N)) for s in range(SEL,N,50)]
BASELINES={'qwen':[('bench','rudder_clone','--attn sdpa'),('bench','grains_clone','--attn eager'),('vti','vtiws_v0.9_t0.9','')],
           'gemma':[('bench','rudder_clone',''),('bench','grains_clone',''),('bench','vti','')]}
def cand(model):
    """CHAIR, AMBER and THRONE settings of the paper at the model's layer, with the old V_obj and with the colour-fixed one."""
    n=16 if model=='qwen' else 24;A=f"--attn {'sdpa' if model=='qwen' else 'eager'} --layer {n}";V=f'--vobj {R}/outputs/rbd-20261030/collect/{model}-vobj2cf.json'
    return {f's{n}a2bl1p05':f'{A} --alpha 2 --lam 1 --pen 0.5',f's{n}a1bl1p05':f'{A} --alpha 1 --lam 1 --pen 0.5',
            'mainMed':f'{A} --alpha 1 --lam 1 --pen 0.5 --norm median',f's{n}a1bl2p1':f'{A} --alpha 1 --lam 2 --pen 1',
            'cf_a2bl1p05':f'{A} --alpha 2 --lam 1 --pen 0.5 {V}','cf_a1bl2p1':f'{A} --alpha 1 --lam 2 --pen 1 {V}','cf_a3bl2p1':f'{A} --alpha 3 --lam 2 --pen 1 {V}'}
def vddbase_args(model):return f"--attn {'sdpa' if model=='qwen' else 'eager'}"
def base_args(model):return '--attn sdpa' if model=='qwen' else ''
def line(m,k,t,s,e,a=''):return f'{m} {k} {t} {s} {e} {a}'.rstrip()
def write_queue(name,lines):
    p=O/'queues'/f'{RUN}_{name}.txt';p.parent.mkdir(exist_ok=True);p.write_text('\n'.join(lines)+'\n',newline='\n');print(p.name,len(lines),'units')

def fetch(name):
    for d in (R/'data/hallucination-suite/images/coco',O/'coco2014/val2014'):
        if (d/name).exists():return d/name
    p=O/'coco2014/val2014'/name;p.parent.mkdir(parents=True,exist_ok=True)
    data=urllib.request.urlopen(f'http://images.cocodataset.org/val2014/{name}',timeout=60).read()
    assert data[:2]==b'\xff\xd8',name;tmp=p.with_suffix(f'.part{os.getpid()}');tmp.write_bytes(data);tmp.replace(p);return p

def make_set():
    t=read(O/'sets/chair_test150.jsonl')[0];by={}
    for s in json.loads(HAT_JSON.read_text(encoding='utf-8')):
        g=by.setdefault(s['image_path'],{'references':s['references'],'samples':[]});g['samples'].append(s['sample_id']);assert g['references']==s['references'],s['image_path']
    assert len(by)==N,len(by)
    with cf.ThreadPoolExecutor(8) as ex:paths=dict(zip(by,ex.map(fetch,by)))
    rows=[]
    for name,g in by.items():
        i=int(name.split('_')[-1].split('.')[0]);p=paths[name]
        rows.append({**t,'id':f'hat:{i}','group_id':f'hat:{i}','image':str(p),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'prompt':os.environ.get('HAT_PROMPT',t['prompt']),'max_new_tokens':int(os.environ.get('HAT_MAX_TOK',t['max_new_tokens'])),
                     'metadata':{'image_id':i,'image_path':name,'hat_samples':g['samples'],'references':g['references']}})
    SET.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    print(SET.name,len(rows),'images, prompt',repr(rows[0]['prompt']),'max_new_tokens',rows[0]['max_new_tokens'])

def queue_dev(models):
    q=[]
    for m in models:
        q+=[line(m,'bench','base',0,SEL,base_args(m)),line(m,'vdd','vddbase',0,SEL,vddbase_args(m))]+[line(m,'vdd',t,0,SEL,a) for t,a in cand(m).items()]
    write_queue('dev',q)

def pick(model,js):
    r=json.loads(Path(js).read_text())['results'];C=cand(model);missing=[t for t in ('base','vddbase') if t not in r]
    assert not missing,f'{model}: no complete {missing} on the selection rows'
    b,v=r['base'],r['vddbase'];c={t:r[t] for t in C if t in r}
    if len(c)<len(C):print(f'{model}: candidates without complete rows, left out: {sorted(set(C)-set(c))}')
    elig={t:x for t,x in c.items() if x['ALOHa']>b['ALOHa'] and x['ALOHa']>v['ALOHa'] and x['RefCov']>=v['RefCov']-2}
    H.mkdir(parents=True,exist_ok=True)
    if not elig:
        (H/f'pick-{model}.txt').write_text('none\n')
        print(f"{model}: picked none: best ALOHa {max((x['ALOHa'] for x in c.values()),default=float('nan')):.1f}, bench Base {b['ALOHa']:.1f}, vdd Base {v['ALOHa']:.1f} (RefCov floor {v['RefCov']-2:.1f}); RBD not continued");return
    top=max(x['ALOHa'] for x in elig.values());near=[t for t,x in elig.items() if x['ALOHa']>=top-.5];p=max(near,key=lambda t:elig[t]['RefCov'])
    (H/f'pick-{model}.txt').write_text(p+'\n')
    print(f"{model}: picked {p} ALOHa {elig[p]['ALOHa']:.1f} RefCov {elig[p]['RefCov']:.1f} (bench Base {b['ALOHa']:.1f}, vdd Base {v['ALOHa']:.1f}/{v['RefCov']:.1f}); eligible {list(elig)}; within 0.5: {near}")

def queue_test(models):
    q=[]
    for s,e in SPANS:
        for m in models:
            f=H/f'pick-{m}.txt';p=f.read_text().strip() if f.exists() else 'none'
            if s>=SEL:
                if p!='none':q.append(line(m,'vdd',p,s,e,cand(m)[p]))
                q+=[line(m,'bench','base',s,e,base_args(m)),line(m,'vdd','vddbase',s,e,vddbase_args(m))]
            q+=[line(m,k,t,s,e,a) for k,t,a in BASELINES[m]]
    write_queue('test',q)

def queue_bg(models):
    q=[]
    for m in models:
        q+=[line(m,'bench','base',0,SEL,base_args(m)),line(m,'vdd','vddbase',0,SEL,vddbase_args(m))]+[line(m,'vdd',t,0,SEL,a) for t,a in cand(m).items()]
        for s,e in SPANS:
            if s>=SEL:q+=[line(m,'bench','base',s,e,base_args(m)),line(m,'vdd','vddbase',s,e,vddbase_args(m))]
            q+=[line(m,k,t,s,e,a) for k,t,a in BASELINES[m]]
    write_queue('bg',q)

def status(models):
    print(f"{'model':7s}{'method':24s}{'rows':>10s}{'s/img':>7s}{'left_h':>8s}");total=0
    for m in models:
        tags={t:N for k,t,_ in BASELINES[m]};tags.update(base=N,vddbase=N)
        f=H/f'pick-{m}.txt';p=f.read_text().strip() if f.exists() else None
        if p and p!='none':tags[p]=N
        for t in cand(m):tags.setdefault(t,SEL)
        for t,n in tags.items():
            rows=[r for g in rundir(m).glob(f'{t}-*-*.jsonl') for r in read(g)];d=len(rows);sec=sum(r.get('elapsed_seconds',0) for r in rows)/max(d,1);left=(n-d)*sec/3600;total+=left
            print(f'{m:7s}{t:24s}{d:6d}/{n:<3d}{sec:7.0f}{left:8.1f}')
    print(f'remaining {total:.1f} GPU-hours (methods with no rows yet count 0); divide by the number of workers')

PILOT={'qwen':('chair_test500_all',['base','s16a2bl1p05']),'gemma':('gchair_test500_all',['base','s24a1bl1p05'])}
def pilot(models):
    caps=collections.defaultdict(list)
    for a in json.loads((R/'data/hallucination-suite/annotations/captions_val2014.json').read_text(encoding='utf-8'))['annotations']:caps[a['image_id']].append(a['caption'])
    rows=[{**r,'metadata':{**r['metadata'],'references':caps[r['metadata']['image_id']]}} for r in read(O/'sets/chair_test500.jsonl')];n=len(rows)
    assert all(r['metadata']['references'] for r in rows);SET.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    for m in models:
        src,tags=PILOT[m];rundir(m).mkdir(parents=True,exist_ok=True)
        for t in tags:shutil.copy(O/'runs'/src/f'{t}-0-{n}.jsonl',rundir(m))
        print(m,'pilot runs',tags,'from',src,n,'rows')

if __name__=='__main__':
    c=sys.argv[1];a=sys.argv[2:]
    if c=='set':make_set()
    elif c=='pilot':pilot(a)
    elif c=='queue_dev':queue_dev(a)
    elif c=='pick':pick(a[0],a[1])
    elif c=='queue_test':queue_test(a)
    elif c=='queue_bg':queue_bg(a)
    else:status(a or ['qwen','gemma'])
