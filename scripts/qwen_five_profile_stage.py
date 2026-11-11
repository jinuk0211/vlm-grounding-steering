"""Route preserved Qwen fit/extraction code to the five-profile protocol."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

from qwen_five_profile_api import ROOT,GPU2,GPU3,load_qwen,require_gemma_complete,verify_run_protocol

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def fp16_preflight(output):
    import torch
    from vlm_grounding.cli import _seed
    from author_clone_bridge import load_authors,verify_clone_sources
    from qwen_author_compat import grains_on_qwen_layer_device
    state={'stage':'loading_model','status':'running','pid':os.getpid(),'started_at':time.time(),'dtype':'float16'}
    file=output/'fp16-preflight.json';save(file,state)
    try:
        require_gemma_complete();protocol=verify_run_protocol(output)
        assert verify_clone_sources()==protocol['author_sources']
        config=json.loads((output/'config.json').read_text());config['model']['dtype']='float16';_seed(config)
        adapter=load_qwen(config['model'],output,'fp16-preflight')
        ready=json.loads((output/'artifacts-ready.json').read_text());assert ready['status']=='verified_complete'
        assert sha(output/'grains-author.pt')==ready['methods']['grains_clone']['sha256']
        artifact=torch.load(output/'grains-author.pt',map_location='cpu',weights_only=True)
        assert artifact['model']=={**config['model'],'dtype':'float32'} and not artifact['gemma_vectors_reused']
        vector=artifact['vector'].to(adapter.device);assert vector.dtype==torch.float32 and torch.isfinite(vector).all()
        record=json.loads((output/'data/fit50.jsonl').read_text().splitlines()[0])
        batch=adapter.prepare(record['image'],'Please describe this image in detail.')
        seen=[]
        def finite(module,args,value):
            tensor=value[0] if isinstance(value,tuple) else value
            assert torch.isfinite(tensor).all();seen.append(str(tensor.dtype))
        handle=adapter.layers[27].register_forward_hook(finite)
        with torch.no_grad():
            result=adapter.model(**batch);assert torch.isfinite(result.logits).all()
        handle.remove();del result
        assert seen and set(seen)=={'torch.float16'}
        diag={};authors=load_authors();_seed(config)
        with grains_on_qwen_layer_device(adapter,vector,authors,diag):answer=adapter.generate(batch,max_new_tokens=32)
        assert answer.strip() and diag['calls']>0 and diag['changed_elements']>0
        state.update(stage='complete',status='passed',activation_dtypes=sorted(set(seen)),native_logits_finite=True,
            steering_vector_dtype=str(vector.dtype),diagnostics=diag,response=answer,benchmark_rows_written=0,completed_at=time.time())
    except BaseException:state.update(stage='failed',status='failed',error=traceback.format_exc());raise
    finally:save(file,state)

def main(output,stage):
    if stage in ['preflight_bf16','preflight_fp32']:
        import preflight_qwen_authors as module
        module.require_gemma_complete=require_gemma_complete;module.load_qwen=load_qwen
        # Original development preflight can use either authorized empty GPU.
        if stage=='preflight_bf16':
            assert os.environ['CUDA_VISIBLE_DEVICES'] in [GPU2,GPU3]
            module.GPU2=os.environ['CUDA_VISIBLE_DEVICES']
        else:
            # Keep the original endpoint/IG20/PCA checks, omit unneeded FP32
            # Base and GrAInS generations. Actual GrAInS inference is FP16.
            tree=ast.parse(inspect.getsource(module.main))
            branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
                isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='phase')
            nodes=branch.orelse
            vector_index=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vector' for t in n.targets))
            cut=next(i for i,n in enumerate(nodes) if i>vector_index and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='batch' for t in n.targets))
            update=nodes[-1]
            assert isinstance(update,ast.Expr) and isinstance(update.value,ast.Call)
            assert isinstance(update.value.func,ast.Attribute) and update.value.func.attr=='update'
            update.value.keywords=[kw for kw in update.value.keywords if kw.arg!='grains_diagnostics']
            branch.orelse=nodes[:cut]+[update]
            ast.fix_missing_locations(tree)
            exec(compile(tree,str(Path(module.__file__))+'::ig_pca_only','exec'),module.__dict__)
        module.main(output,'bf16' if stage=='preflight_bf16' else 'fp32')
        if stage=='preflight_fp32':
            path=output/'fp32-preflight.json';proof=json.loads(path.read_text())
            proof['execution_override']={'role':'Original endpoint, IG20 and PCA checks only',
                'fp32_base_or_grains_generations':0,'stage_source_sha256':sha(Path(__file__))}
            save(path,proof)
    elif stage in ['fit_revis','fit_vti']:
        import fit_qwen_retained as module
        module.require_gemma_complete=require_gemma_complete;module.verify_run_protocol=verify_run_protocol;module.load_qwen=load_qwen
        module.main(output,stage.split('_',1)[1])
    elif stage=='extract':
        import extract_qwen_author_grains as module
        module.require_gemma_complete=require_gemma_complete;module.verify_run_protocol=verify_run_protocol;module.load_qwen=load_qwen
        module.main(output)
    elif stage=='preflight_fp16':fp16_preflight(output)
    else:raise ValueError(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',required=True)
    a=p.parse_args();main(a.output,a.stage)
