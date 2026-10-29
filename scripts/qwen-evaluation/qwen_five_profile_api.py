"""Native Qwen loader for five profiles: BF16 methods and FP16 GrAInS."""
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from vlm_grounding.backend import ModelAdapter,load_adapter as native_load_adapter

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
GEMMA=ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507'
CHECKPOINT='Qwen/Qwen2.5-VL-7B-Instruct'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'
GPU2='GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3='GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR=GPU2+','+GPU3

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_gemma_complete():
    state=json.loads((GEMMA/'status.json').read_text())
    assert state['stage']=='complete','Finish all five Gemma profiles and official audit before Qwen GPU work'
    report=json.loads((GEMMA/'report.json').read_text())
    assert report['status']=='verified_complete' and report['audit']['all_17500_responses_verified']
    assert report['audit']['official_scorers_used'] and report['audit']['all_3500_grains_responses_fresh_fp16']
    assert report['audit']['no_base_fp32_generated_or_included']
    return report

def verify_run_protocol(output):
    assert output.resolve().parent==ROOT/'outputs'
    p=json.loads((output/'protocol.json').read_text());body={k:v for k,v in p.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['sha256']
    assert p['model']['checkpoint']==CHECKPOINT and p['model']['revision']==REVISION
    assert p['model']['family']=='qwen2_5_vl' and p['model']['dtype']=='bfloat16' and p['model']['local_files_only']
    assert p['methods']==['base','revis','vti','rudder_clone','grains_clone'] and p['method_dtypes']['grains_clone']=='float16'
    for file,digest in p['source_hashes'].items():assert sha(ROOT/file)==digest,file
    for file,digest in p['manifest_hashes'].items():assert sha(output/'data'/file)==digest,file
    assert sha(output/'config.json')==p['config_sha256'] and sha(output/'tasks.json')==p['tasks_sha256']
    return p

def load_qwen(config,output=None,worker=None):
    require_gemma_complete()
    assert config['family']=='qwen2_5_vl' and config['checkpoint']==CHECKPOINT and config['revision']==REVISION
    assert config.get('local_files_only')
    visible=os.environ['CUDA_VISIBLE_DEVICES'];dtype=config['dtype']
    if dtype in ['bfloat16','float16']:
        assert visible in [GPU2,GPU3] and torch.cuda.device_count()==1
        adapter=native_load_adapter(config)
    else:
        assert dtype=='float32' and visible==PAIR and torch.cuda.device_count()==2
        shared=dict(revision=REVISION,local_files_only=True,trust_remote_code=False)
        processor=AutoProcessor.from_pretrained(CHECKPOINT,**shared)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(CHECKPOINT,torch_dtype=torch.float32,
            attn_implementation=config.get('attn_implementation','eager'),device_map='balanced',
            max_memory={0:'21GiB',1:'21GiB'},**shared)
        adapter=ModelAdapter(model,processor,config)
        assert {str(p.device) for p in model.parameters()}=={'cuda:0','cuda:1'}
        assert {p.dtype for p in model.parameters() if p.is_floating_point()}=={torch.float32}
    assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
    cached=try_to_load_from_cache(CHECKPOINT,'config.json',revision=REVISION)
    assert isinstance(cached,str) and Path(cached).parent.name==REVISION
    observed=getattr(adapter.model.config,'_commit_hash',None);assert observed in [None,REVISION]
    adapter.model.config._commit_hash=REVISION
    proof={'checkpoint':CHECKPOINT,'revision':REVISION,'cached_config':cached,'observed_revision_metadata':observed,
        'restored_revision_metadata':REVISION,'visible_devices':visible,'dtype':dtype,
        'parameter_devices':sorted({str(p.device) for p in adapter.model.parameters()}),
        'parameter_dtypes':sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
        'hf_device_map':{k:str(v) for k,v in getattr(adapter.model,'hf_device_map',{}).items()},
        'quantization':False,'cpu_or_disk_offload':False,'model_weights_changed':False,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'role':'IG extraction only' if dtype=='float32' else 'benchmark inference',
        'steering_vector_dtype':'torch.float32' if dtype=='float16' else None}
    assert str(getattr(torch,dtype)) in proof['parameter_dtypes']
    if dtype=='float16':assert 'torch.bfloat16' not in proof['parameter_dtypes']
    adapter.qwen_gpu23_device_proof=proof
    if output is not None:(output/(worker+'-device-map.json')).write_text(json.dumps(proof,indent=2)+'\n')
    return adapter
