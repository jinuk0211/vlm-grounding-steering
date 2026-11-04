"""Original GrAInS functions, fresh FP16 inference on GPU2 or GPU3 only."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
from author_clone_bridge import ROOT, load_authors, verify_clone_sources, grains_context
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed

GPU2='GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3='GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verified_protocol(output):
    p=json.loads((output/'protocol.json').read_text());body={k:v for k,v in p.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['sha256']
    assert p['methods']==['base','revis','vti','rudder_clone','grains_clone']
    assert p['method_dtypes']['grains_clone']=='float16' and 'base_fp32' not in p['methods']
    for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['manifest_hashes'].items():assert sha(output/'data'/name)==digest,name
    assert sha(output/'config.json')==p['config_sha256']
    assert sha(output/'tasks.json')==p['tasks_sha256']
    assert verify_clone_sources()==p['author_sources']
    assert sha(Path(p['grains_artifact']))==p['grains_artifact_sha256']
    return p

def load(output,worker):
    protocol=verified_protocol(output)
    visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible in [GPU2,GPU3] and torch.cuda.device_count()==1
    config=json.loads((output/'config.json').read_text());config['model']['dtype']='float16'
    _seed(config);adapter=load_adapter(config['model'])
    from huggingface_hub import try_to_load_from_cache
    cached=try_to_load_from_cache(config['model']['checkpoint'],'config.json',revision=config['model']['revision'])
    assert isinstance(cached,str) and Path(cached).parent.name==config['model']['revision']
    observed=getattr(adapter.model.config,'_commit_hash',None);assert observed in [None,config['model']['revision']]
    adapter.model.config._commit_hash=config['model']['revision']
    proof={'visible_devices':visible,'dtype':'float16','revision':config['model']['revision'],
           'parameter_dtypes':sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
           'parameter_devices':sorted({str(p.device) for p in adapter.model.parameters()}),
           'quantization':False,'cpu_or_disk_offload':False,'cached_config':cached}
    assert 'torch.float16' in proof['parameter_dtypes'] and 'torch.bfloat16' not in proof['parameter_dtypes']
    assert proof['parameter_devices']==['cuda:0']
    save(output/(worker+'-device-map.json'),proof)
    artifact=torch.load(protocol['grains_artifact'],map_location='cpu',weights_only=True)
    source_model=dict(artifact['model']);source_model['dtype']='float16'
    assert source_model==config['model']
    vector=artifact['vector'].to(adapter.device)
    assert vector.dtype==torch.float32, 'Preserve original author vector dtype; do not underflow the 1e-8 epsilon'
    assert torch.isfinite(vector).all() and vector.norm()>0
    proof['steering_vector_dtype']=str(vector.dtype)
    save(output/(worker+'-device-map.json'),proof)
    return protocol,config,adapter,vector,load_authors()

def main(output,worker,preflight):
    state={'pid':os.getpid(),'worker':worker,'completed_records':0,'dtype':'float16','started_at':time.time()}
    status=output/(('preflight' if preflight else 'worker-'+worker)+'.json')
    def update(**kw):state.update(kw,updated_at=time.time());save(status,state)
    try:
        update(stage='loading_model');protocol,config,adapter,vector,authors=load(output,worker)
        if preflight:
            record=read(output/'data/fit50.jsonl')[0]
            batch=adapter.prepare(record['image'],'Please describe this image in detail.')
            # Check the actual native decoder activations and logits before committing benchmarks.
            seen={'calls':0,'dtypes':set()}
            def finite_hook(module,args,output_value):
                value=output_value[0] if isinstance(output_value,tuple) else output_value
                assert torch.isfinite(value).all(),'Nonfinite native FP16 hidden state'
                seen['calls']+=1;seen['dtypes'].add(str(value.dtype))
            handle=adapter.layers[27].register_forward_hook(finite_hook)
            with torch.no_grad():
                logits=adapter.model(**batch).logits
                assert torch.isfinite(logits).all(),'Nonfinite native FP16 logits'
            handle.remove();assert seen['calls']>0 and seen['dtypes']=={'torch.float16'}
            diagnostics={};_seed(config)
            with grains_context(adapter,vector,authors,diagnostics):response=adapter.generate(batch,max_new_tokens=32)
            assert response.strip() and diagnostics['calls']>0 and diagnostics['changed_elements']>0
            update(stage='complete',status='passed',response=response,diagnostics=diagnostics,
                   activation_dtypes=sorted(seen['dtypes']),native_logits_finite=True,
                   extraction_dtype='float32',inference_dtype='float16',benchmark_rows_written=0)
            return
        assert json.loads((output/'preflight.json').read_text())['status']=='passed'
        datasets={b:read(output/'data'/(b+'.jsonl')) for b in ['pope','chair']}
        update(stage='ready')
        for task in json.loads((output/'tasks.json').read_text()):
            if task['method']!='grains_clone':continue
            lease=output/'leases'/task['id'];claim=output/'claims'/task['id'];path=output/'predictions'/(task['id']+'.jsonl')
            if (claim/'complete.json').exists():continue
            try:lease.mkdir()
            except FileExistsError:continue
            claim.mkdir(exist_ok=True)
            rows=datasets[task['benchmark']][task['start']:task['stop']]
            assert not path.exists(),'Existing prefix requires an explicit continuation lease amendment'
            save(claim/'owner.json',{'worker':worker,'pid':os.getpid(),'started_at':time.time()})
            update(stage='generating',method='grains_clone',benchmark=task['benchmark'],task=task['id'],task_completed=0)
            with path.open('x') as sink:
                for index,record in enumerate(rows):
                    assert sha(record['image'])==record['image_sha256'];_seed(config)
                    batch=adapter.prepare(record['image'],record['prompt']);diagnostics={}
                    torch.cuda.synchronize();start=time.perf_counter()
                    with grains_context(adapter,vector,authors,diagnostics):
                        response=adapter.generate(batch,max_new_tokens=record['max_new_tokens'])
                    torch.cuda.synchronize();assert diagnostics['calls']>0
                    assert response.strip() and 0<adapter.last_generated_tokens<=record['max_new_tokens']
                    row={**record,'method':'grains_clone','prediction':response,'generated_tokens':adapter.last_generated_tokens,
                         'elapsed_seconds':time.perf_counter()-start,'protocol_sha256':protocol['sha256'],
                         'dtype':'float16','worker':worker,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
                         'author_diagnostics':diagnostics,'artifact_sha256':protocol['grains_artifact_sha256']}
                    sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush();del batch
                    update(completed_records=state['completed_records']+1,task_completed=index+1,last_seconds=row['elapsed_seconds'])
            save(claim/'complete.json',{'worker':worker,'records':len(rows),'sha256':sha(path),'completed_at':time.time()})
        assert verify_clone_sources()==protocol['author_sources'];update(stage='complete',completed_at=time.time())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',required=True);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();main(a.output,a.worker,a.preflight)
