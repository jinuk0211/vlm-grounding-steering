"""THRONE (amazon-science/THRONE, commit 45badbb7) on throne_test60, run with /workspace/venv-throne.
Steps 2 and 3 of scripts/one-click.sh, unchanged: abstractive QA with google/flan-t5-{base,large,xl} (pinned revisions,
batch 32, M=3 question formats), then throne_score_aqa.py (micro and classwise, voting thresholds 5 8 9). Our captions
replace step 1 (prompt "Describe this image in detail.", throne_generate.GENERATION_PROMPT).
Headline: micro (image-level) P, R, F1, F0.5 at threshold 9 (the scorer's default, unanimous 3 judges x 3 formats);
paired bootstrap vs base (2000, seed 0) on the same per-image sets the official functions produce.
usage: score_throne.py GPU RUNDIR"""
import glob,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np

BENCH=Path(os.environ.get('BENCH','/workspace/bench'))  # project-b: /home/project-b/LJW/bench
T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
JUDGES=[str(BENCH/f'models/flan-t5-{s}') for s in ('base','large','xl')]
SETS=Path(os.environ.get('RBD2_SETS','/workspace/sets'))
TSET=os.environ.get('RBD2_THRONE_SET','throne_test60');N=int(os.environ.get('RBD2_N','60'))
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def judge(gpu,d):
    (d/'combined').mkdir(exist_ok=True)
    env={**os.environ,'LOCAL_RANK':'0','WORLD_SIZE':'1','RANK':'0','CUDA_VISIBLE_DEVICES':gpu}
    for j in JUDGES:
        name=f"evaluations_{j.replace('/','_')}.json"
        if (d/'combined'/name).exists():continue
        out=subprocess.run([sys.executable,'throne_aqa_evaluation.py','--coco_file',COCO,'--response_file',str(d/'responses.json'),
                            '--evaluator_model_path',j,'--per_device_batch_size','32','--save_path',str(d/'evaluations.json')],
                           cwd=T,env=env,capture_output=True,text=True)
        assert out.returncode==0,out.stderr[-3000:]
        shutil.move(d/name,d/'combined'/name)

def per_image(d,thr=9):
    sys.path.insert(0,str(T));import throne_score_aqa as S,pycocotools.coco as coco,contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
    cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats}
    pred=S.prepare_aqa_preds(glob.glob(str(d/'combined')+'/*.json'),idx);col,dis=S.majority_vote_collapsing(pred,voting_threshold=thr)
    P={};R={}
    for k in col:   # micro_avergaing_metrics, per image
        g=set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k])));dq=set(dis[k])
        gp={x for x in g if x not in dq};pp={x for x in col[k] if x not in dq}
        P[k],R[k]=(len(gp&pp)/len(pp),len(gp&pp)/len(gp)) if pp and gp else (0.0,0.0)
    return P,R

def summ(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):
        f1=np.nanmean(2*p*r/(p+r));f05=np.nanmean(1.25*p*r/(0.25*p+r))
    return {'P':100*p.mean(),'R':100*r.mean(),'F1':100*f1,'F0.5':100*f05}

def main(gpu,rundir):
    rundir=Path(rundir);refs=read(SETS/f'{TSET}.jsonl');ids=[r['id'] for r in refs];res={};per={}
    for f in sorted(glob.glob(str(rundir/f'*-0-{N}.jsonl'))):
        tag=Path(f).name[:-len(f'-0-{N}.jsonl')];rows={r['id']:r for r in read(f)}
        if not all(i in rows for i in ids):continue
        d=rundir/'eval'/tag;d.mkdir(parents=True,exist_ok=True)
        resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
        if not (d/'responses.json').exists() or json.loads((d/'responses.json').read_text())!=resp:
            shutil.rmtree(d/'combined',ignore_errors=True);(d/'responses.json').write_text(json.dumps(resp))
        judge(gpu,d)
        off=''.join(subprocess.run([sys.executable,'throne_score_aqa.py','--coco_val_ann_path',COCO,'--model_eval_path',str(d/'combined'),
                                    '--thresholds','5','8','9','--metric_strategy',s],cwd=T,capture_output=True,text=True,check=True).stdout for s in ('micro','class'))
        (d/'official_scores.txt').write_text(off)
        P,R=per_image(d);keys=[r['metadata']['image_id'] for r in refs]
        per[tag]=(np.array([P[k] for k in keys]),np.array([R[k] for k in keys]))
        res[tag]={**summ(*per[tag]),'tokens':float(np.mean([rows[i]['generated_tokens'] for i in ids]))}
    assert 'base' in per,'no complete base run'
    rng=np.random.default_rng(0);B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)];bp,br=per['base']
    for t,(p,r) in per.items():
        if t=='base':continue
        d=np.array([[summ(p[k],r[k])[q]-summ(bp[k],br[k])[q] for q in ('P','R','F0.5')] for k in B])
        res[t]['ci']={q:[float(np.quantile(d[:,j],.025)),float(np.quantile(d[:,j],.975))] for j,q in enumerate(('P','R','F0.5'))}
    (rundir/'score.json').write_text(json.dumps({'set':TSET,'n':len(ids),'threshold':9,'averaging':'micro','results':res},indent=1))
    for t,v in res.items():
        ci=' '.join(f"d{q}[{c[0]:+.1f},{c[1]:+.1f}]" for q,c in v.get('ci',{}).items())
        print(f"{t:28s} P {v['P']:5.1f} R {v['R']:5.1f} F1 {v['F1']:5.1f} F0.5 {v['F0.5']:5.1f}  tok {v['tokens']:4.0f}  {ci}")

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
