"""One comparison directory per set, runs/<set>_test150_cmp/<tag>-0-150.jsonl, so every method is scored against the same
Base (rbd2 base: BF16, sdpa) on the same 150 images:
  rbd2 tags    copied from runs/<set>_test150_all (pb_join150.py); THRONE judge outputs (eval/<tag>) are copied too
  baselines    joined from runs/<set>_test150_bl (baseline_bench.py: REVIS/RUDDER BF16 + sdpa, GrAInS/VTI FP16 + eager
               with FP32 attention scores); CHAIR GrAInS/VTI reuse the first 150 rows of the verified test run
               (original-vti-gpu1-final, the same settings; their FP16 captions have no ' addCriterion').
Every piece must match sets/<set>_test150.jsonl id for id."""
import glob,json,re,shutil,sys
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';N=150
OLD=R/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair'
BL=('revis','rudder_clone','grains_clone','vti_author')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def write(p,rows):p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
bad=[]
for s in ('chair','amber','throne'):
    ref=[r['id'] for r in read(O/f'sets/{s}_test150.jsonl')];src=O/f'runs/{s}_test150_all';out=O/f'runs/{s}_test150_cmp'
    out.mkdir(exist_ok=True)
    for f in sorted(src.glob(f'*-0-{N}.jsonl')):
        shutil.copy(f,out/f.name)
        if (src/'eval'/f.name[:-len(f'-0-{N}.jsonl')]).is_dir():
            shutil.copytree(src/'eval'/f.name[:-len(f'-0-{N}.jsonl')],out/'eval'/f.name[:-len(f'-0-{N}.jsonl')],dirs_exist_ok=True)
    for m in BL:
        if s=='chair' and m in ('grains_clone','vti_author'):
            rows=read(OLD/f'{m}.jsonl')[:N];src_=f'original-vti-gpu1-final/chair/{m}.jsonl[:150]'
        else:
            piece={}
            for f in sorted(glob.glob(str(O/f'runs/{s}_test150_bl/{m}-*-*.jsonl'))):
                mm=re.fullmatch(re.escape(m)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
                if not mm:continue
                a,b=int(mm[1]),int(mm[2]);r=read(f)
                if len(r)==b-a and [x['id'] for x in r]==ref[a:b]:piece.setdefault(a,(b,r,Path(f).name))
            rows=[];at=0;names=[]
            while at<N and at in piece:b,r,n=piece[at];rows+=r;names.append(n);at=b
            src_=' + '.join(names)
        if [x['id'] for x in rows]!=ref:bad.append(f'{s} {m}: {len(rows)} rows');continue
        write(out/f'{m}-0-{N}.jsonl',rows)
        print(f'{s:7s}{m:13s}{src_}  addCriterion {sum("addCriterion" in x["prediction"] for x in rows)}')
if bad:print('INCOMPLETE',*bad,sep='\n');sys.exit(1)
