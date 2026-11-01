"""Which precision setting removes the ' addCriterion' glitch of Qwen2.5-VL bf16 Base captions?
Variants: attn (eager|sdpa), vision tower dtype (bf16|fp32). Greedy, 512 tokens, first N CHAIR test images."""
import argparse,json,os,time
import torch
import vlm_grounding.backend as B
import read_gap_collect_v2 as C

def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES']
    cfg=dict(json.loads((C.MODELS['qwen']/'config.json').read_text())['model'],attn_implementation=a.attn)
    ad=B.load_adapter(cfg);tok=ad.processor.tokenizer
    if a.vision=='fp32':ad.model.model.visual.float()
    bad=0
    for r in C.read('/workspace/chair_test100.jsonl')[:a.n]:
        b=ad.prepare(r['image'],r['prompt']);pl=b['input_ids'].shape[1];t0=time.perf_counter()
        with torch.no_grad():o=ad.model.generate(**b,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=512)
        text=tok.decode(o[0,pl:],skip_special_tokens=True);g='addCriterion' in text;bad+=g
        print(r['id'],o.shape[1]-pl,'tok',f'{time.perf_counter()-t0:.0f}s','GLITCH' if g else 'ok','|',text[:110].replace('\n',' '),flush=True)
    print(f'{a.attn} vision-{a.vision}: glitch {bad}/{a.n}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attn',default='eager');p.add_argument('--vision',default='bf16')
    p.add_argument('--n',type=int,default=12);main(p.parse_args())
