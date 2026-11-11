L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R/scripts/rbd2
export BENCH=$L/bench RBD2_SETS=$O/sets PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000
$L/venvs/throne/bin/python - <<'PY'
import glob,hashlib,json,sys,contextlib,io
from pathlib import Path
import numpy as np
import score_throne as S
sys.path.insert(0,str(S.T));import throne_score_aqa as A,pycocotools.coco as coco
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(S.COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats}
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
keys=[r['metadata']['image_id'] for r in S.read(O/'sets/throne_full1000.jsonl')];ids=[r['id'] for r in S.read(O/'sets/throne_full1000.jsonl')]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def sha(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()
def per(files,thr):
    pred=A.prepare_aqa_preds(files,idx);col,dis=A.majority_vote_collapsing(pred,voting_threshold=thr);P=[];Rr=[];c={'pred_empty':0,'gt_empty':0,'no_overlap':0}
    for k in keys:
        dq=set(dis[k]);gp={x for x in gt[k] if x not in dq};pp={x for x in col[k] if x not in dq}
        if not pp:c['pred_empty']+=1
        if not gp:c['gt_empty']+=1
        if pp and gp and not (gp&pp):c['no_overlap']+=1
        p,r=(len(gp&pp)/len(pp),len(gp&pp)/len(gp)) if pp and gp else (0.0,0.0);P.append(p);Rr.append(r)
    return np.array(P),np.array(Rr),c
rng=np.random.default_rng(0);B=[rng.integers(0,1000,1000) for _ in range(2000)]
for name,run,diag,tags in [('Qwen2.5-VL-7B','throne_full1000','throne_full1000_2judges',['base','s16a1bl2p1','revis','rudder_clone','grains_clone','vti_author']),
                           ('Gemma-4-E4B','gemma_throne1000','gemma_throne1000_2judges',['base','s24a1bl2p1','revis','rudder_clone','grains_clone','vti'])]:
    print(f'##### {name}  runs/{run}  (2-judge files: diag/{diag}/<tag>/combined)')
    res={}
    for t in tags:
        rows={r['id']:r for r in S.read(O/'runs'/run/f'{t}-0-1000.jsonl')};assert all(i in rows for i in ids)
        tok=[rows[i]['generated_tokens'] for i in ids]
        fs=sorted(f for f in glob.glob(str(O/'diag'/diag/t/'combined'/'*.json')));assert len(fs)==2,fs
        same=all(sha(f)==sha(O/'runs'/run/'eval'/t/'combined'/Path(f).name) for f in fs)
        p,r,c=per(fs,6);res[t]=(p,r)
        v=S.summ(p,r);pr0=int(((p==0)&(r==0)).sum())
        line=f"{t:13s} n {len(ids)}  P {v['P']:5.1f} R {v['R']:5.1f} F1 {v['F1']:5.1f} F0.5 {v['F0.5']:5.1f}  tok {np.mean(tok):5.1f} @512 {sum(x>=512 for x in tok):3d}  P=R=0 {pr0:3d} (pred empty {c['pred_empty']}, gt empty {c['gt_empty']}, no overlap {c['no_overlap']})  diag==eval {same}"
        for j in fs:
            pj,rj,_=per([j],3);vj=S.summ(pj,rj);line+=f"\n    judge {Path(j).name.split('flan-t5-')[1][:-5]:5s} (3 formats, unanimous thr 3): P {vj['P']:5.1f} R {vj['R']:5.1f} F1 {vj['F1']:5.1f} F0.5 {vj['F0.5']:5.1f}  P=R=0 {int(((pj==0)&(rj==0)).sum())}"
        print(line)
    bp,br=res['base']
    for t,(p,r) in res.items():
        if t=='base':continue
        d=np.array([[S.summ(p[k],r[k])[q]-S.summ(bp[k],br[k])[q] for q in ('P','R','F0.5')] for k in B])
        print(f"  CI vs bench Base  {t:13s} "+' '.join(f"d{q}[{np.quantile(d[:,j],.025):+.1f},{np.quantile(d[:,j],.975):+.1f}]" for j,q in enumerate(('P','R','F0.5'))))
PY
for f in full1000_ours.log score_throne_full.log score500.log gemma_ours_fix.log score_vtiws100.log; do echo "##### RAW logs/$f"; cat $O/logs/$f | grep -v 'it/s'; done
echo "##### RAW logs/gemma.log (final tables)"; sed -n '/gemma_throne1000 baselines/,$p' $O/logs/gemma.log
echo "##### score.json files"; ls $O/runs/*/score*.json 2>/dev/null
