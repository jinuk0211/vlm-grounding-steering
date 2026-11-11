"""Mode direction d_l = mean h_l["describe appearance/arrangement"] - mean h_l["list every object"] at the last
prompt position, every layer (index 0 = embeddings, l = output of decoder layer l). COCO train2014 probe images
that are not in the dev split; never the CHAIR test images."""
import argparse,json,os
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from rbd_common import ROOT

PROMPTS={'describe':'Describe the appearance and arrangement of what you see in this image in detail.',
         'list':'List every object in this image.'}

def main(a):
    a.output.mkdir(parents=True,exist_ok=True);assert os.environ['CUDA_VISIBLE_DEVICES']
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model'];adapter=load_adapter(config)
    dev={json.loads(l)['metadata']['image_id'] for l in (ROOT/'data/chair-dev-train2014-100/chair.jsonl').read_text().splitlines()}
    imgs=[r for r in C.read(ROOT/'data/coco-train2014-probe5000/list.jsonl') if r['image_id'] not in dev][:a.n]
    acc={k:None for k in PROMPTS}
    with torch.no_grad():
        for r in imgs:
            for k,p in PROMPTS.items():
                b=adapter.prepare(r['image'],p);hs=adapter.model(**b,output_hidden_states=True,use_cache=False,return_dict=True).hidden_states
                h=torch.stack([x[0,-1].float() for x in hs]);acc[k]=h if acc[k] is None else acc[k]+h
    d=(acc['describe']-acc['list'])/len(imgs);mean_norm=((acc['describe']+acc['list'])/(2*len(imgs))).norm(dim=-1)
    torch.save({'d':d.cpu(),'mean_state_norm':mean_norm.cpu(),'n':len(imgs),'prompts':PROMPTS},a.output/f'{a.model}-dvec.pt')
    print('saved',d.shape,'|d|/|h| per layer:',[round(float(x),3) for x in (d.norm(dim=-1)/mean_norm)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=200);main(p.parse_args())
