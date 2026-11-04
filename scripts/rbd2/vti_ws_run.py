"""VTI (author code, alpha 0.9 visual and textual) with the whitespace-normalised textual direction of vti_fix.py
(outputs/rbd2-20261102/vti_fix/vti-author-ws.pt) on a caption set. Same model setup and row format as
baseline_bench.py vti_author (FP16 weights, eager attention with FP32 scores, greedy, seed 42 per image); rows go to
OUTPUT/TAG-START-END.jsonl and are resumed. --av/--at scale the visual/textual direction (author default 0.9; 0 = not
installed), for the CHAIR alpha choice on chair_dev60 (vti_chair.sh).
usage: vti_ws_run.py --set SET.jsonl --output DIR --start 0 --end 1000 [--av 0.9 --at 0.9 --tag vti_author_ws]"""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from qwen_fp16_attention_guard import stable_fp16_attention
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed
import run_qwen_original_vti as original

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
Q=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
ART=ROOT/'outputs/rbd2-20261102/vti_fix/vti-author-ws.pt'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    config=json.loads((Q/'config.json').read_text());cfg=dict(config['model'],dtype='float16',attn_implementation='eager')
    refs=read(a.set)[a.start:a.end];a.output.mkdir(parents=True,exist_ok=True);TAG=a.tag
    name=f'{TAG}-{a.start}-{a.end}';rows_file=a.output/f'{name}.jsonl';status=a.output/f'{name}-status.json'
    done=read(rows_file) if rows_file.exists() else []
    for r,ref in zip(done,refs):assert r['id']==ref['id'] and r['method']==TAG
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),total=len(refs),done=len(done),artifact=str(ART),artifact_sha256=sha(ART),
           alpha_visual=a.av,alpha_textual=a.at)
    attention={}
    try:
        with stable_fp16_attention(attention):
            adapter=load_adapter(cfg);assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
            assert getattr(adapter.model.config,'_commit_hash',None) in (None,REVISION);adapter.model.config._commit_hash=REVISION
            _,layers=original.imports();art=torch.load(ART,map_location='cpu',weights_only=True)
            assert art['author_commit']==original.COMMIT and art['author_source_hashes']==original.verify_author()
            assert art['demo_whitespace_normalized']
            if a.av:original.install(adapter.vision_layers,art['visual_direction'][1:],a.av,layers,vision=True)
            if a.at:original.install(adapter.layers,art['textual_direction'][1:],a.at,layers)
            update(stage='running',gpu=torch.cuda.get_device_name(0))
            with rows_file.open('a') as f:
                for i,ref in enumerate(refs[len(done):],start=len(done)):
                    assert sha(ref['image'])==ref['image_sha256'],ref['image']
                    _seed(config);t0=time.perf_counter()
                    text=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=int(ref['max_new_tokens']))
                    f.write(json.dumps({**ref,'method':TAG,'prediction':text,'generated_tokens':adapter.last_generated_tokens,
                        'elapsed_seconds':time.perf_counter()-t0,'dtype':'float16','attn_implementation':'eager',
                        'attention_score_dtype':'float32','gpu':torch.cuda.get_device_name(0),'artifact_sha256':sha(ART),
                        'alpha_visual':a.av,'alpha_textual':a.at})+'\n');f.flush()
                    update(done=i+1)
        update(stage='complete',fp16_attention_calls=attention.get('calls'))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=1000)
    p.add_argument('--av',type=float,default=.9);p.add_argument('--at',type=float,default=.9);p.add_argument('--tag',default='vti_author_ws')
    main(p.parse_args())
