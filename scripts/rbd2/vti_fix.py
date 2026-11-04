"""VTI (author code) on Qwen2.5-VL with a whitespace-normalised demo file.

Diagnosis (2026-11-03): the author's textual direction on Qwen is a newline direction. Logit lens of
textual_direction[24..28] puts '\\n', '.\\n', ' \\n' ... at the top of 152k tokens, and ' -' at layer 14, which matches
the '\\n- \\n- ...' loops in 970/1000 full-THRONE captions. Cause: in experiments/data/hallucination_vti_demos.jsonl,
88/100 'value' captions contain newlines and only 19/100 'h_value' captions do, so value - h_value at the last token
carries the paragraph format.

Fix: same author pipeline (get_demos with seed 42 and 70 demos, prompt_pairs, obtain_textual_vti with the visual VTI
active, PCA rank 1, alpha 0.9), but value and h_value have their whitespace collapsed to single spaces. The visual
direction does not depend on the captions and is reused from the existing artifact.

Steps: re-extract the textual direction -> logit lens of old and new direction -> generate N images of a caption set
with the new artifact (both directions, alpha 0.9) -> rows + summary.
usage: vti_fix.py --set SET.jsonl --out DIR [--n 20]"""
import argparse,contextlib,hashlib,json,random,sys,time,types
from pathlib import Path
import torch
from qwen_fp16_attention_guard import stable_fp16_attention
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed
import run_qwen_original_vti as original

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
Q=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
OLD=ROOT/'outputs/qwen-original-vti-1793260357'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def lens(adapter,tok,direction,layers=(14,20,24,26,27,28)):
    W=adapter.model.lm_head.weight.float();g=adapter.model.model.language_model.norm.weight.float() \
        if hasattr(adapter.model.model,'language_model') else adapter.model.model.norm.weight.float()
    nl={i:repr(tok.decode([i])) for i in (198,271,481,715)}
    for L in layers:
        d=direction[L].float().to(W.device);d=d/d.norm();z=W@(g*d)
        top=[tok.decode([i]) for i in torch.topk(z,12).indices.tolist()]
        print(f'  L{L} top {top} | newline/dash ranks',{nl[i]:int((z>z[i]).sum()) for i in nl},flush=True)

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    utils,layers=original.imports()
    old=torch.load(OLD/'vti-author.pt',map_location='cpu',weights_only=True)
    assert old['author_commit']==original.COMMIT and old['author_source_hashes']==original.verify_author()
    demos=Path(original.AUTHOR)/'experiments/data/hallucination_vti_demos.jsonl'
    rows=read(demos);ws=a.out/'hallucination_vti_demos.ws.jsonl'
    ws.write_text(''.join(json.dumps({**r,'value':' '.join(r['value'].split()),'h_value':' '.join(r['h_value'].split())})+'\n' for r in rows))
    print('demos',len(rows),'newline in value/h_value before',sum('\n' in r['value'] for r in rows),sum('\n' in r['h_value'] for r in rows),
          'after',sum('\n' in r['value'] for r in read(ws)),sum('\n' in r['h_value'] for r in read(ws)),flush=True)
    config=json.loads((Q/'config.json').read_text())
    cfg=dict(config['model'],dtype='float16',attn_implementation='eager');attention={}
    with stable_fp16_attention(attention):
        adapter=load_adapter(cfg);assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
        assert getattr(adapter.model.config,'_commit_hash',None) in (None,REVISION);adapter.model.config._commit_hash=REVISION
        tok=adapter.processor.tokenizer
        print('logit lens, OLD textual direction:',flush=True);lens(adapter,tok,old['textual_direction'])
        original.install(adapter.vision_layers,old['visual_direction'][1:].to(adapter.device),.9,layers,vision=True)
        # Same as original.fit: seed 42 right before get_demos; num_trials only sizes the unused masked images.
        args=types.SimpleNamespace(num_demos=70,num_trials=1,mask_ratio=.99,data_file=str(OLD/'data'))
        picked=[]
        def pairs(*p,**k):picked.extend(d['image'] for d in p[3]);return original.prompt_pairs(adapter,*p,**k)
        saved=utils.get_prompts;utils.get_prompts=pairs;random.seed(42)
        try:_,inputs=utils.get_demos(args,original.ImageProcessor(adapter.processor.image_processor),adapter.model,None,file_path=str(ws))
        finally:utils.get_prompts=saved
        manifest=OLD/'demo-manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text());names=json.dumps(m)
            print('demo selection matches original manifest:',all(p in names for p in picked),len(picked),flush=True)
        t0=time.time();textual,_=utils.obtain_textual_vti(original.TextCapture(adapter),inputs,None,rank=1)
        assert torch.isfinite(textual).all() and textual.shape==old['textual_direction'].shape
        cos=torch.nn.functional.cosine_similarity(textual.float(),old['textual_direction'].float(),dim=-1)
        print(f'textual direction re-extracted in {time.time()-t0:.0f}s; cos(new, old) per layer:',[round(float(c),2) for c in cos],flush=True)
        art={**old,'textual_direction':textual,'demo_whitespace_normalized':True,'demo_file_sha256':sha(ws),'demo_images':picked}
        torch.save(art,a.out/'vti-author-ws.pt')
        print('logit lens, NEW textual direction:',flush=True);lens(adapter,tok,textual)
        original.install(adapter.layers,textual[1:].to(adapter.device),.9,layers)
        refs=read(a.set)[:a.n];out=a.out/f'vti_ws-0-{a.n}.jsonl'
        with out.open('w') as f:
            for ref in refs:
                _seed(config);t=time.perf_counter()
                text=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=int(ref['max_new_tokens']))
                n=adapter.last_generated_tokens
                f.write(json.dumps({**ref,'method':'vti_author_ws','prediction':text,'generated_tokens':n,'elapsed_seconds':time.perf_counter()-t,
                                    'dtype':'float16','attn_implementation':'eager','attention_score_dtype':'float32'})+'\n');f.flush()
                print(f"{ref['id']} tok {n} | {text[:160]!r}",flush=True)
    r=read(out);tk=[int(x['generated_tokens']) for x in r]
    print(f'SUMMARY n {len(r)} mean tok {sum(tk)/len(tk):.0f} at512 {sum(t>=510 for t in tk)} '
          f"dash-loops {sum(x['prediction'].count(chr(10)+'-')>20 for x in r)}",flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=20);main(p.parse_args())
