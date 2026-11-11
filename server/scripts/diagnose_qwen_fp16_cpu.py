"""Development-only CPU diagnostic; never emits benchmark predictions."""
import json,os,time,traceback
from pathlib import Path
import torch
from qwen_gpu0_migration_api import ROOT,sha,save,verify_protocol
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(8);torch.set_num_interop_threads(2)
    output=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    verify_protocol(output)
    target=output/'fp16-cpu-diagnostic.json'
    state={'pid':os.getpid(),'stage':'loading_cpu_model','started_at':time.time(),'source_sha256':sha(__file__),
        'benchmark_predictions_written':0,'device':'cpu','dtype':'float16','threads':8}
    def update(**kw):state.update(kw,updated_at=time.time());save(target,state)
    update()
    try:
        config=json.loads((output/'config.json').read_text());config['model'].update(dtype='float16',device='cpu');_seed(config)
        adapter=load_adapter(config['model'])
        bad=[name for name,p in adapter.model.named_parameters() if not torch.isfinite(p).all()]
        update(stage='forward_original_fp16',nonfinite_parameters=bad);assert not bad
        row=json.loads((output/'data/fit50.jsonl').read_text().splitlines()[0])
        batch=adapter.prepare(row['image'],'Please describe this image in detail.')
        def tensors(value):
            if torch.is_tensor(value):yield value
            elif isinstance(value,(tuple,list)):
                for v in value:yield from tensors(v)
            elif isinstance(value,dict):
                for v in value.values():yield from tensors(v)
        def summary(t):
            finite=torch.isfinite(t);v=t[finite]
            return {'dtype':str(t.dtype),'shape':list(t.shape),'nonfinite':int((~finite).sum()),
                'max_abs_finite':float(v.abs().max()) if v.numel() else None}
        def hook(name):
            def check(module,args,value):
                for t in tensors(value):
                    if t.is_floating_point() and not torch.isfinite(t).all():
                        update(stage='first_nonfinite_located',first_module=name,module_class=type(module).__name__,
                            inputs=[summary(v) for v in tensors(args)],output=summary(t))
                        raise FloatingPointError(name)
            return check
        handles=[module.register_forward_hook(hook(name)) for name,module in adapter.model.named_modules() if not list(module.children())]
        try:
            with torch.no_grad():result=adapter.model(**batch)
            update(stage='native_cpu_fp16_finite',logits_finite=bool(torch.isfinite(result.logits).all()))
        except FloatingPointError:pass
        finally:
            for h in handles:h.remove()
        update(completed_at=time.time())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
