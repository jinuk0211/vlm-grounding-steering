"""RTX 2080 Ti has no bf16 tensor cores: time Base decoding (rbd_decode's lam=0 path: greedy, model generation
config, 512 tokens) in a given dtype on the first CHAIR test images and compare with the stored A5000 bf16 Base
captions when available (Qwen: first 50 test images)."""
import argparse,json,os,time
from pathlib import Path
import torch
import vlm_grounding.backend as B
import read_gap_collect_v2 as C

def main(a):
    if a.dtype=='float32':
        import fp32_two_gpu;fp32_two_gpu.install()
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    cfg=dict(json.loads((C.MODELS[a.model]/'config.json').read_text())['model'],dtype=a.dtype)
    refs=C.read('/workspace/chair_test100.jsonl')[:a.n]
    stored={r['image_id']:r['caption'] for r in json.loads(Path(a.ref).read_text())} if a.ref else {}
    ad=B.load_adapter(cfg);tok=ad.processor.tokenizer;same=0;pref=[]
    for r in refs:
        b=ad.prepare(r['image'],r['prompt']);pl=b['input_ids'].shape[1];t0=time.perf_counter()
        with torch.no_grad():o=ad.model.generate(**b,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=512)
        dt=time.perf_counter()-t0;ids=o[0,pl:];text=tok.decode(ids,skip_special_tokens=True)
        line=f"{r['id']} {len(ids)} tok {dt:.1f}s {len(ids)/dt:.1f} tok/s"
        s=stored.get(r['metadata']['image_id'])
        if s is not None:
            w1,w2=text.split(),s.split();k=next((i for i,(x,y) in enumerate(zip(w1,w2)) if x!=y),min(len(w1),len(w2)))
            same+=text.strip()==s.strip();pref.append(k/max(len(w2),1));line+=f" | identical {text.strip()==s.strip()} common-prefix {k}/{len(w2)} words"
        print(line,'|',text[:150].replace('\n',' '),'|| stored:',(s or '')[:150].replace('\n',' '),flush=True)
    if pref:print(f'{a.dtype}: identical {same}/{len(pref)}, mean common-prefix fraction {sum(pref)/len(pref):.2f}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dtype',required=True)
    p.add_argument('--n',type=int,default=5);p.add_argument('--ref');main(p.parse_args())
