"""Qwen author-VTI alpha sweep (user request 2026-10-30: alpha 0.45 / 0.225).

Same model load, author artifact, original install() and VTILayer as
qwen_vti_chair_gpu23.py; the only change is the alpha passed to install()
(0.9 in the completed run). Reads the existing VTI artifact read-only and
writes only under the sweep folder given by --output.
"""
import argparse
import json
import os
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

GPU = {1: 'GPU-5cf44493-6a32-f174-b98a-c8bc5702ee73',
       2: 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde',
       3: 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}
VTI = ROOT/'outputs/qwen-original-vti-1793260357'
ALPHAS = (0.45, 0.225)


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().split()[21])


def main(out, index, alpha, bench, start, end):
    assert index in GPU and os.environ['CUDA_VISIBLE_DEVICES']==GPU[index]
    assert alpha in ALPHAS and bench in ('pope','chair') and 0<=start<end
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.manual_seed(42)
    require_gemma_complete()
    parent_protocol=verify_protocol(original.PARENT)
    manifest=json.loads((VTI/'protocol.json').read_text())
    artifact_path=VTI/'vti-author.pt';artifact_digest=sha(artifact_path)
    refs=original.read(original.PARENT/'data'/(bench+'.jsonl'))
    assert len(refs)=={'pope':3000,'chair':500}[bench] and end<=len(refs)
    target=out/f'alpha{alpha}';target.mkdir(parents=True,exist_ok=True)
    name=f'{bench}-{start}-{end}'
    rows_file=target/f'{name}.jsonl';state_file=target/f'{name}-status.json'
    # Resume only an exact, newline-terminated prefix of this file's own rows.
    prefix=original.read(rows_file) if rows_file.exists() else []
    assert not prefix or rows_file.read_bytes().endswith(b'\n')
    for row,i in zip(prefix,range(start,end)):
        assert row['id']==refs[i]['id'] and row['vti_alpha']==alpha
    state={'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'gpu_index':index,'gpu_uuid':GPU[index],
           'alpha':alpha,'bench':bench,'range':[start,end],'stage':'loading',
           'completed_records':len(prefix),'resumed_from':len(prefix),'expected_records':end-start,
           'source_sha256':sha(__file__),'original_worker_sha256':sha(original.__file__),
           'artifact_sha256':artifact_digest,'parent_protocol_sha256':parent_protocol['sha256']}
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
            dtypes=sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()})
            assert sorted({str(p.device) for p in adapter.model.parameters()})==['cuda:0']
            assert 'torch.float16' in dtypes and 'torch.bfloat16' not in dtypes
            update(gpu_name=torch.cuda.get_device_name(0))
            artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
            assert artifact['author_commit']==original.COMMIT
            assert artifact['author_source_hashes']==original.verify_author()
            original.install(adapter.vision_layers,artifact['visual_direction'][1:],alpha,layers,vision=True)
            original.install(adapter.layers,artifact['textual_direction'][1:],alpha,layers)
            diagnostics={};handles=[]
            for mname,module in adapter.model.named_modules():
                if isinstance(module,layers.VTILayer):
                    kind='vision' if '.visual.' in mname else 'text'
                    def observe(_module,args,value,kind=kind):
                        assert torch.isfinite(value).all()
                        d=diagnostics.setdefault(kind,{'calls':0,'changed_elements':0,'max_abs_delta':0.0})
                        diff=(value.float()-args[0].float()).abs()
                        d['calls']+=1;d['changed_elements']+=int((diff>0).sum())
                        d['max_abs_delta']=max(d['max_abs_delta'],float(diff.max()))
                    handles.append(module.register_forward_hook(observe))
            assert len(handles)==len(adapter.layers)+len(adapter.vision_layers)
            update(stage='generating')
            with rows_file.open('a' if prefix else 'x') as sink:
                for i in range(start+len(prefix),end):
                    ref=refs[i];assert sha(ref['image'])==ref['image_sha256']
                    diagnostics.clear();began=time.perf_counter()
                    response=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=ref['max_new_tokens'])
                    assert all(diagnostics.get(k,{}).get('changed_elements',0)>0 for k in ('vision','text'))
                    row={**ref,'method':f'vti_author_a{alpha}','vti_alpha':alpha,'prediction':response,
                         'generated_tokens':adapter.last_generated_tokens,
                         'elapsed_seconds':time.perf_counter()-began,'gpu_uuid':GPU[index],
                         'dtype':'float16','attention_score_dtype':'float32',
                         'checkpoint':config['checkpoint'],'revision':config['revision'],
                         'protocol_sha256':manifest['sha256'],'parent_protocol_sha256':manifest['parent_protocol_sha256'],
                         'author_commit':original.COMMIT,'artifact_sha256':artifact_digest,
                         'sweep_source_sha256':state['source_sha256'],
                         'original_hook_file':layers.VTILayer.forward.__code__.co_filename,
                         'original_vti_diagnostics':dict(diagnostics),
                         'attention_amendment_sha256':sha(original.PARENT/'fp16-attention-amendment.json')}
                    sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush()
                    update(completed_records=state['completed_records']+1,last_index=i)
            for handle in handles:handle.remove()
        assert [r['id'] for r in original.read(rows_file)]==[refs[i]['id'] for i in range(start,end)]
        update(stage='complete',output_sha256=sha(rows_file),completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-index',type=int,required=True)
    p.add_argument('--alpha',type=float,required=True);p.add_argument('--bench',required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    a=p.parse_args();main(a.output,a.gpu_index,a.alpha,a.bench,a.start,a.end)
