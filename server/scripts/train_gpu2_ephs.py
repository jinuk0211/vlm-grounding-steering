"""Train the native EPHS method on the designated single A5000 GPU."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed, _provenance, _validate_artifact, _versions
from vlm_grounding.method import discover, fit, steer

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
GPU='GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main(output,resume_selection=False):
    output=Path(output);status=output/'training-status.json';state={'pid':os.getpid(),'started_at':time.time(),'forward_calls':0,'gpu_uuid':GPU}
    def update(**v):state.update(v,updated_at=time.time());save(status,state)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and torch.cuda.device_count()==1
        protocol=json.loads((output/'training-protocol.json').read_text());config=protocol['model']
        for p,d in protocol['source_hashes'].items():assert sha(ROOT/p)==d,p
        for p,d in protocol['development_input_hashes'].items():assert sha(p)==d,p
        _seed(config);update(stage='loading_model');adapter=load_adapter(config['model']);adapter.model.config._commit_hash=config['model']['revision']
        forward=adapter.forward
        def observed_forward(batch):
            state['forward_calls']+=1
            if state['forward_calls']%20==0:update()
            return forward(batch)
        adapter.forward=observed_forward
        discovery=read(output/'data/discovery.jsonl');fitting=read(output/'data/fit.jsonl')
        assert {r['group_id'] for r in discovery}.isdisjoint(r['group_id'] for r in fitting)
        update(stage='discovering_heads',candidate_layers=config['discovery']['layers'],heads_per_layer=adapter.num_heads,discovery_images=len(discovery),versions=_versions())
        started=time.time()
        if resume_selection:
            selection=json.loads((output/'selection.json').read_text())
            _validate_artifact(selection,config['model'],adapter=adapter)
            assert selection['_provenance']['manifests'][0]['sha256']==sha(output/'data/discovery.jsonl')
        else:
            selection=discover(adapter,discovery,config['discovery'])
            selection['_provenance']=_provenance(adapter,config,[str(output/'data/discovery.jsonl')],discovery)
            save(output/'selection.json',selection)
        update(stage='fitting_gains',discovery_seconds=time.time()-started,resumed_selection=resume_selection,selected_heads=selection['selected_heads'],accepted_discovery_images=len(selection['discovery_ids']))
        # Only saved autograd intermediates move to CPU. Model math and parameters are unchanged.
        # This keeps gradient fitting within the single 24GB GPU's memory budget.
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            artifact=fit(adapter,fitting,selection,config['fit'])
        artifact['_provenance']=_provenance(adapter,config,[str(output/'data/fit.jsonl')],fitting,selection['_provenance'])
        artifact['_provenance']['selection_sha256']=sha(output/'selection.json')
        values=[float(e['delta']) for e in artifact['selected_heads']]
        assert all(torch.isfinite(torch.tensor(values))) and any(abs(v)>1e-6 for v in values),values
        save(output/'ephs.json',artifact);_validate_artifact(artifact,config['model'],adapter=adapter)
        probe=fitting[0];batch=adapter.prepare(probe['image'],probe['prompt']);_seed(config)
        base=adapter.generate(batch,max_new_tokens=16)
        with steer(adapter,artifact,config['methods']['ephs']):
            pred=adapter.generate(batch,max_new_tokens=16)
        validation={'status':'passed','finite_nonzero_gains':True,'bounded_gains':all(abs(x)<=artifact['max_gain'] for x in values),'probe_base':base,'probe_ephs':pred,'fit_steps':artifact['fit_steps'],'gain_deltas':values,'max_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'activation_storage':'torch.autograd.graph.save_on_cpu(pin_memory=False)'}
        save(output/'ephs-validation.json',validation);update(stage='complete',artifact=str(output/'ephs.json'),validation=validation,completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--resume-selection',action='store_true');args=parser.parse_args();main(args.output,args.resume_selection)
