"""Explicit GPU0 hardware amendment around the sealed five-profile protocol."""
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from vlm_grounding.backend import load_adapter
from qwen_five_profile_api import ROOT,CHECKPOINT,REVISION,require_gemma_complete
from qwen_five_profile_api import verify_run_protocol as verify_parent

GPU0='GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify_protocol(output):
    p=verify_parent(output);a=json.loads((output/'gpu0-hardware-amendment.json').read_text())
    assert a['parent_protocol_sha256']==p['sha256'] and a['user_requested_gpu_indices']==[0]
    for name,digest in a['source_hashes'].items():assert sha(ROOT/name)==digest,name
    stop=Path(a['stop_proof']);assert sha(stop)==a['stop_proof_sha256']
    proof=json.loads(stop.read_text());assert proof['all_our_selected_instances_exited']
    journal=(output/'extraction.jsonl').read_bytes()[:proof['committed_journal_bytes']]
    assert hashlib.sha256(journal).hexdigest()==proof['journal_prefix_sha256']
    return p

def load_gpu0(config,output,worker):
    require_gemma_complete();verify_protocol(output)
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU0 and torch.cuda.device_count()==1
    assert config['checkpoint']==CHECKPOINT and config['revision']==REVISION and config['family']=='qwen2_5_vl'
    assert config['local_files_only'] and config['dtype'] in ['float32','bfloat16','float16']
    cached=try_to_load_from_cache(CHECKPOINT,'config.json',revision=REVISION)
    assert isinstance(cached,str) and Path(cached).parent.name==REVISION
    adapter=load_adapter(config)
    observed=getattr(adapter.model.config,'_commit_hash',None);assert observed in [None,REVISION]
    adapter.model.config._commit_hash=REVISION
    assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
    proof={'checkpoint':CHECKPOINT,'revision':REVISION,'cached_config':cached,'observed_revision_metadata':observed,
        'restored_revision_metadata':REVISION,'visible_devices':GPU0,'physical_gpu_index':0,'dtype':config['dtype'],
        'parameter_devices':sorted({str(p.device) for p in adapter.model.parameters()}),
        'parameter_dtypes':sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
        'quantization':False,'cpu_or_disk_offload':False,'model_weights_changed':False,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),'gpu_name':torch.cuda.get_device_name(0),
        'steering_vector_dtype':'torch.float32' if config['dtype']=='float16' else None,
        'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json')}
    assert proof['parameter_devices']==['cuda:0'] and str(getattr(torch,config['dtype'])) in proof['parameter_dtypes']
    if config['dtype']=='float32':assert proof['parameter_dtypes']==['torch.float32']
    if config['dtype']=='float16':assert 'torch.bfloat16' not in proof['parameter_dtypes']
    adapter.qwen_gpu23_device_proof=proof;save(output/(worker+'-device-map.json'),proof)
    return adapter
