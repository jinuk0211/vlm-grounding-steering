"""CPU-only prefix diagnostic; never modifies the running benchmark or its model.

Measure literal author RUDDER before/after Gemma's attention output RMSNorm.
Use disjoint development images, without inspecting test labels or tuning.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
from author_clone_bridge import ROOT, NativeBatchProcessor, load_authors, verify_clone_sources
import torch
from vlm_grounding.backend import load_adapter


class PrefixDone(Exception):pass
class PrefixModel:
    def __init__(self, model):self.native=model
    def __getattr__(self,name):return getattr(self.native,name)
    def __call__(self,*args,**kwargs):
        try:return self.native(*args,**kwargs)
        except PrefixDone:return None


def main():
    output=ROOT/'outputs/author-rudder-scale-diagnostic';output.mkdir(exist_ok=False)
    state={'pid':os.getpid(),'device':'cpu','started_at':time.time(),'completed':0}
    def update(**kw):
        state.update(kw,updated_at=time.time());temp=output/'status.tmp';temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temp.replace(output/'status.json')
    try:
        import psutil
        available=psutil.virtual_memory().available
        assert available>45*1024**3,'Insufficient free CPU memory for independent diagnostic'
        supported=getattr(torch.cpu,'_is_avx512_bf16_supported',lambda:False)()
        config=json.loads((ROOT/'configs/gemma4_e4b.local.json').read_text())
        config['model'].update(device='cpu',dtype='bfloat16' if supported else 'float32')
        torch.set_num_threads(4);update(stage='loading_cpu_model',dtype=config['model']['dtype'],available_ram_gib=available/1024**3)
        adapter=load_adapter(config['model']);authors=load_authors();sources=verify_clone_sources()
        target=adapter.layers[1].self_attn;norm=adapter.layers[1].post_attention_layernorm
        prefix=PrefixModel(adapter.model)
        fit=[json.loads(x) for x in (ROOT/'data/mmvp-full/fit.jsonl').read_text().splitlines()]
        results=[]
        for index in [0,5,10,15,20,25]:
            record=fit[index];update(stage='measuring',record_id=record['id'])
            batch=adapter.prepare(record['image'],record['prompt'])
            snapshot={}
            def capture_residual(_module,args,kwargs):snapshot['residual']=(args[0] if args else kwargs['hidden_states']).detach().clone()
            def capture_attention(_module,args,output):snapshot['attention']=output[0].detach().clone()
            def stop(_module,args):raise PrefixDone()
            handles=[adapter.layers[1].register_forward_pre_hook(capture_residual,with_kwargs=True),
                     target.register_forward_hook(capture_attention),adapter.layers[2].register_forward_pre_hook(stop)]
            try:
                with torch.no_grad():
                    card=authors.rudder.compute_card_vector_batch(prefix,NativeBatchProcessor(batch),images=[None],questions=['native Gemma prompt'],layer_idx=1,pooling='attn',local=False)
            finally:
                for handle in handles:handle.remove()
            h=snapshot['attention'];residual=snapshot['residual'][:,-1:]
            hook=authors.rudder.BayesianGatingHookMaskedDynamic(max_alpha=8.0,sensitivity=5.0,concentration=1.0,
                carrier=authors.rudder.mask_carrier,clamp=(0.05,1.0),rms_match=False,record=True)
            authors.rudder.mask_carrier.set(authors.generation_mask(batch));hook.set_vector(card)
            with torch.no_grad():
                changed=hook(target,(),(h.clone(),None))[0]
                before=h[:,-1:];after=changed[:,-1:]
                before_norm=norm(before);after_norm=norm(after)
                delta=(after.float()-before.float());norm_delta=after_norm.float()-before_norm.float()
                merged_before=residual+before_norm;merged_after=residual+after_norm
                item={'id':record['id'],'dtype':config['model']['dtype'],'attention_norm':float(before.float().norm()),
                    'additive_delta_norm':float(delta.norm()),'raw_relative_change':float(delta.norm()/before.float().norm()),
                    'normed_relative_change':float(norm_delta.norm()/before_norm.float().norm()),
                    'residual_relative_change':float((merged_after.float()-merged_before.float()).norm()/merged_before.float().norm()),
                    'card_cosine_with_attention':float(torch.nn.functional.cosine_similarity(card.float(),before[0].float(),dim=-1)[0]),
                    'gate':float(hook.fetch_last_stats()['gate'][0,-1,0]),
                    'raw_changed_fraction':float((before!=after).float().mean()),
                    'normed_changed_fraction':float((before_norm!=after_norm).float().mean())}
            authors.rudder.mask_carrier.clear();results.append(item)
            update(completed=len(results),last=item);print(json.dumps(item),flush=True)
        summary={key:statistics.median(r[key] for r in results) for key in ['raw_relative_change','normed_relative_change','residual_relative_change','attention_norm','additive_delta_norm','card_cosine_with_attention','gate','raw_changed_fraction','normed_changed_fraction']}
        report={'status':'complete','scope':'Six fixed development inputs, CPU prefix only; no test tuning or benchmark changes',
             'model':config['model'],'layer':1,'alpha':8.0,'results':results,'medians':summary,
             'author_sources':sources,'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'limitation':'CPU activations are not a direct recording of the running GPU model; final answer logits are not measured'}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        update(stage='complete',medians=summary)
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
