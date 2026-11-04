L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R/scripts/rbd2
export BENCH=$L/bench RBD2_SETS=$O/sets PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 CUDA_VISIBLE_DEVICES=
$L/venvs/throne/bin/python - <<'PY'
import glob,re,sys,contextlib,io
from pathlib import Path
import score_throne as S
sys.path.insert(0,str(S.T));import throne_score_aqa as A,pycocotools.coco as coco
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(S.COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats}
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
refs=S.read(O/'sets/throne_full1000.jsonl');keys=[r['metadata']['image_id'] for r in refs]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
FRUIT=re.compile(r'\boranges\b|\borange (slice|slices|fruit|fruits|wedge|wedges|segment|segments|peel|half|halves)\b|\b(an|one) orange\b(?! (shirt|jacket|vest|hat|cap|top|dress|cone|cones|light|lights|color|colour|hue|tint|sky|glow|ball|stripe|stripes|and|or|,))',re.I)
for name,run,diag,ours in [('Qwen2.5-VL-7B','throne_full1000','throne_full1000_2judges','s16a1bl2p1'),('Gemma-4-E4B','gemma_throne1000','gemma_throne1000_2judges','s24a1bl2p1')]:
    print('\n##### %s: class "orange" (COCO fruit), 2 judges, threshold 6'%name)
    print('  images with orange in COCO gt: %d'%sum('orange' in gt[k] for k in keys))
    for t in ('base',ours):
        rows={r['id']:r for r in S.read(O/'runs'/run/('%s-0-1000.jsonl'%t))};cap={r['metadata']['image_id']:rows[r['id']]['prediction'] for r in refs}
        fs=sorted(glob.glob(str(O/'diag'/diag/t/'combined'/'*.json')));col,dis=A.majority_vote_collapsing(A.prepare_aqa_preds(fs,idx),voting_threshold=6)
        fp=[k for k in keys if 'orange' in col[k] and 'orange' not in dis[k] and 'orange' not in gt[k]]
        word=[k for k in keys if re.search(r'\borange',cap[k],re.I)]
        fpw=[k for k in fp if re.search(r'\borange',cap[k],re.I)];fpf=[k for k in fp if FRUIT.search(cap[k])]
        print('  %-11s captions containing "orange*": %3d | orange FPs: %3d, of which caption contains "orange*": %3d, looks like the fruit (oranges / orange slice / an orange ...): %3d'%(t,len(word),len(fp),len(fpw),len(fpf)))
        for k in fp[:3]:
            m=re.search(r'.{0,60}\borange\w*.{0,40}',cap[k],re.I|re.S);print('      e.g. image %d: ...%s...'%(k,m[0].replace('\n',' ') if m else '(no "orange" in caption)'))
PY
