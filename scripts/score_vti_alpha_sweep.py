"""Score a VTI alpha sweep slice against the verified five-method rows on the same items.

Reuses the official scorers exactly as audit_qwen_vti_replacement.score does
(cached unmodified CHAIR constructor state, POPE official helper). CPU only.
"""
import argparse,json,os,re
from pathlib import Path

from vlm_grounding.evaluation import _load_external_module,attach_chair_scores,evaluate_predictions
from qwen_gpu0_migration_api import ROOT,save,sha

FINAL=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final'
METHODS=['base','revis','rudder_clone','grains_clone','vti_author']

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def words(text):return len(re.findall(r'[A-Za-z]{2,}',text))

def chair_scorer():
    data=ROOT/'data/hallucination-suite';cache=json.loads((data/'chair-scorer-state.json').read_text())
    source=ROOT/'upstream/chair/chair.py';assert cache['built_with_unmodified_constructor'] and sha(source)==cache['scorer_sha256']
    module=_load_external_module(source,'original_chair_alpha_sweep')
    scorer=module.CHAIR.__new__(module.CHAIR);scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    return scorer

def main(sweep,bench,tag):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    sweep=sweep.resolve();out=sweep/f'score-{tag}';out.mkdir()
    runs={}
    for folder in sorted(sweep.glob('alpha*')):
        rows=[r for f in sorted(folder.glob(bench+'-*.jsonl')) for r in read(f)]
        if rows:runs['vti_author_a'+folder.name[5:]]=rows
    ids=[r['id'] for r in next(iter(runs.values()))]
    for name,rows in runs.items():assert sorted(r['id'] for r in rows)==sorted(ids),name
    for m in METHODS:
        by={r['id']:r for r in read(FINAL/bench/(m+'.jsonl'))};runs[m]=[by[i] for i in ids]
    for name in runs:by={r['id']:r for r in runs[name]};runs[name]=[by[i] for i in ids]
    if bench=='pope':
        import audit_hallucination_suite as helper
    else:scorer=chair_scorer()
    result={'bench':bench,'items':len(ids),'ids_sha256_input':[sha(f) for f in sorted(sweep.glob('alpha*/'+bench+'-*.jsonl'))],'metrics':{}}
    for name,part in runs.items():
        if bench=='pope':
            value=helper.pope_metrics(part)
        else:
            path=out/(name+'.captions.json');save(path,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in part])
            detail=scorer.compute_chair(str(path),'image_id','caption');save(out/(name+'.chair.official.json'),detail)
            value=evaluate_predictions(attach_chair_scores(part,detail))
        value={k:v for k,v in value.items() if not isinstance(v,dict)}
        w=sorted(words(r['prediction']) for r in part)
        value.update(at_token_cap=sum(r['generated_tokens']>=r['max_new_tokens'] for r in part),
                     median_words=w[len(w)//2],mean_generated_tokens=sum(r['generated_tokens'] for r in part)/len(part),
                     yes_answers=sum(r['prediction'].strip().lower().startswith('yes') for r in part) if bench=='pope' else None)
        result['metrics'][name]=value
    save(out/'score.json',result);print(json.dumps(result['metrics'],indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sweep',type=Path,required=True)
    p.add_argument('--bench',required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();main(a.sweep,a.bench,a.tag)
