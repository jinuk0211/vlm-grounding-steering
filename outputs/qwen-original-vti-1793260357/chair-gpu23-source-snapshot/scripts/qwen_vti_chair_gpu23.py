"""Disjoint Qwen author-VTI CHAIR continuation on user-authorized GPUs 2 and 3."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from vlm_grounding.backend import load_adapter

import run_qwen_original_vti as original
from qwen_gpu0_migration_api import ROOT, save, sha, verify_protocol
from qwen_fp16_attention_guard import stable_fp16_attention
from qwen_five_profile_api import CHECKPOINT, REVISION, require_gemma_complete

GPU = {2: 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde',
       3: 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}
RANGES = {2: [(60,167),(394,447)], 3: [(227,334),(447,500)]}


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().split()[21])


def main(out, index):
    assert index in GPU and os.environ['CUDA_VISIBLE_DEVICES']==GPU[index]
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.manual_seed(42)
    require_gemma_complete()
    parent_protocol=verify_protocol(original.PARENT)
    manifest=json.loads((out/'protocol.json').read_text())
    amendment=json.loads((out/'chair-gpu23-amendment.json').read_text())
    assert amendment['parent_protocol_sha256']==parent_protocol['sha256']
    assert amendment['vti_protocol_sha256']==manifest['sha256']
    assert amendment['gpu_uuids'][str(index)]==GPU[index]
    assert amendment['ranges'][str(index)]==[list(r) for r in RANGES[index]]
    assert amendment['source_hashes']['scripts/qwen_vti_chair_gpu23.py']==sha(__file__)
    assert amendment['source_hashes']['scripts/run_qwen_original_vti.py']==sha(original.__file__)
    artifact_path=out/'vti-author.pt';artifact_digest=sha(artifact_path)
    assert amendment['artifact_sha256']==artifact_digest
    refs=original.read(original.PARENT/'data/chair.jsonl');assert len(refs)==500
    target=out/'chair-gpu23';target.mkdir(exist_ok=True)
    rows_file=target/f'gpu{index}.jsonl';state_file=target/f'gpu{index}-status.json'
    assert not rows_file.exists()
    state={'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'gpu_index':index,
           'gpu_uuid':GPU[index],'ranges':RANGES[index],'stage':'loading',
           'completed_records':0,'source_sha256':sha(__file__),
           'original_worker_sha256':sha(original.__file__),
           'artifact_sha256':artifact_digest,
           'amendment_sha256':sha(out/'chair-gpu23-amendment.json')}
    def update(**kwargs):
        state.update(kwargs,updated_at=time.time());save(state_file,state)
    update()
    try:
        utils,layers=original.imports();attention={}
        with stable_fp16_attention(attention):
            config=json.loads((original.PARENT/'config.json').read_text())['model'];config['dtype']='float16'
            assert config['checkpoint']==CHECKPOINT and config['revision']==REVISION and config['local_files_only']
            cached=try_to_load_from_cache(CHECKPOINT,'config.json',revision=REVISION)
            assert isinstance(cached,str) and Path(cached).parent.name==REVISION
            adapter=load_adapter(config)
            observed=getattr(adapter.model.config,'_commit_hash',None)
            assert observed in (None,REVISION)
            adapter.model.config._commit_hash=REVISION
            assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
            proof={'checkpoint':CHECKPOINT,'revision':REVISION,'cached_config':cached,
                   'observed_revision_metadata':observed,'restored_revision_metadata':REVISION,
                   'visible_devices':GPU[index],'physical_gpu_index':index,'dtype':'float16',
                   'parameter_devices':sorted({str(p.device) for p in adapter.model.parameters()}),
                   'parameter_dtypes':sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
                   'quantization':False,'cpu_or_disk_offload':False,'model_weights_changed':False,
                   'gpu_name':torch.cuda.get_device_name(0),
                   'hardware_amendment_sha256':sha(out/'chair-gpu23-amendment.json')}
            assert proof['parameter_devices']==['cuda:0']
            assert 'torch.float16' in proof['parameter_dtypes'] and 'torch.bfloat16' not in proof['parameter_dtypes']
            save(target/f'gpu{index}-device-map.json',proof)
            artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
            assert artifact['author_commit']==original.COMMIT
            assert artifact['author_source_hashes']==original.verify_author()
            original.install(adapter.vision_layers,artifact['visual_direction'][1:],.9,layers,vision=True)
            original.install(adapter.layers,artifact['textual_direction'][1:],.9,layers)
            diagnostics={};handles=[]
            for name,module in adapter.model.named_modules():
                if isinstance(module,layers.VTILayer):
                    kind='vision' if '.visual.' in name else 'text'
                    def observe(_module,args,value,kind=kind):
                        assert torch.isfinite(value).all()
                        d=diagnostics.setdefault(kind,{'calls':0,'changed_elements':0,'max_abs_delta':0.0})
                        diff=(value.float()-args[0].float()).abs()
                        d['calls']+=1;d['changed_elements']+=int((diff>0).sum())
                        d['max_abs_delta']=max(d['max_abs_delta'],float(diff.max()))
                    handles.append(module.register_forward_hook(observe))
            assert len(handles)==len(adapter.layers)+len(adapter.vision_layers)
            update(stage='generating')
            with rows_file.open('x') as sink:
                for start,end in RANGES[index]:
                    for i in range(start,end):
                        ref=refs[i];assert sha(ref['image'])==ref['image_sha256']
                        diagnostics.clear();began=time.perf_counter()
                        response=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=ref['max_new_tokens'])
                        assert all(diagnostics.get(k,{}).get('changed_elements',0)>0 for k in ('vision','text'))
                        row={**ref,'method':'vti_author','prediction':response,'generated_tokens':adapter.last_generated_tokens,
                             'elapsed_seconds':time.perf_counter()-began,'gpu_uuid':GPU[index],
                             'gpu23_hardware_amendment_sha256':sha(out/'chair-gpu23-amendment.json'),
                             'dtype':'float16','attention_score_dtype':'float32',
                             'checkpoint':config['checkpoint'],'revision':config['revision'],
                             'protocol_sha256':manifest['sha256'],'parent_protocol_sha256':manifest['parent_protocol_sha256'],
                             'author_commit':original.COMMIT,'artifact_sha256':artifact_digest,
                             'original_hook_file':layers.VTILayer.forward.__code__.co_filename,
                             'original_vti_diagnostics':dict(diagnostics),
                             'attention_amendment_sha256':sha(original.PARENT/'fp16-attention-amendment.json')}
                        sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush()
                        update(completed_records=state['completed_records']+1,last_index=i)
            for handle in handles:handle.remove()
        assert len(original.read(rows_file))==sum(end-start for start,end in RANGES[index])
        update(stage='complete',output_sha256=sha(rows_file),
               device_proof_sha256=sha(target/f'gpu{index}-device-map.json'),completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,required=True)
    args=parser.parse_args();main(args.output,args.gpu_index)
