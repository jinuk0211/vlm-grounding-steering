"""Actual checkpoint preflight for the pinned author-clone Gemma bridge."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from author_clone_bridge import ROOT, load_authors, verify_clone_sources, rudder_context, grains_context, extract_one
import numpy as np
import torch
from vlm_grounding.backend import load_adapter


def main(output):
    output.mkdir(exist_ok=False)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}
    def save(**changes):
        state.update(changes, updated_at=time.time())
        (output / 'status.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    try:
        assert state['gpu'] == 'GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'
        sources = verify_clone_sources()
        (output / 'author-sources.json').write_text(json.dumps(sources, indent=2)+'\n')
        authors = load_authors()
        save(stage='loading_fp32_model')
        config = json.loads((ROOT / 'configs/gemma4_e4b.local.json').read_text())
        config['model']['dtype'] = 'float32'
        adapter = load_adapter(config['model'])
        for parameter in adapter.model.parameters():
            parameter.requires_grad_(False)
        rows = [json.loads(x) for x in (ROOT / 'data/mmvp-full/fit.jsonl').read_text().splitlines()]
        record = rows[0]
        results = {}
        for task, prompt, limit in [('pope',record['prompt'],8), ('chair','Please describe this image in detail.',32)]:
            save(stage='rudder_'+task)
            batch = adapter.prepare(record['image'], prompt)
            diagnostic = {}
            with rudder_context(adapter,batch,task,authors,diagnostic):
                response = adapter.generate(batch,limit)
            assert diagnostic['calls'] > 0 and diagnostic['active_positions'] > 0
            results['rudder_'+task] = {'response':response,'generated_tokens':adapter.last_generated_tokens,'diagnostics':diagnostic}
        save(stage='grains_original_ig20_top50')
        original_ig = authors.grad.apply_integrated_gradients_contrastive
        def observed(*args, **kwargs):
            started = time.time()
            result = original_ig(*args, **kwargs)
            assert all(torch.isfinite(v).all() for v in result)
            torch.save({'positive':result[0].detach().cpu(),'negative':result[1].detach().cpu()},output/'raw_ig.pt')
            results['ig'] = {'seconds':time.time()-started,'steps':kwargs.get('steps',20),
                             'positive_abs_max':float(result[0].abs().max()),'negative_abs_max':float(result[1].abs().max())}
            return result
        authors.grad.apply_integrated_gradients_contrastive = observed
        values = extract_one(adapter,record,authors)
        hidden={k:np.stack([v]) for k,v in values.items() if k.startswith('hidden_')}
        np.savez(output/'one_fit_hidden.npz',**hidden)
        vector=authors.steering.compute_steering_vector(hidden,27,'pca')
        assert torch.isfinite(vector).all() and vector.norm() > 0
        torch.save(vector,output/'one_fit_vector.pt')
        save(stage='grains_original_tuple_hook')
        batch=adapter.prepare(record['image'],'Please describe this image in detail.')
        diag={}
        with grains_context(adapter,vector.to(adapter.device),authors,diag):
            response=adapter.generate(batch,32)
        assert diag['calls'] > 0 and diag['changed_elements'] > 0
        results['grains']={'response':response,'generated_tokens':adapter.last_generated_tokens,'diagnostics':diag,
                           'vector_norm':float(vector.norm())}
        assert verify_clone_sources() == sources
        (output/'result.json').write_text(json.dumps({'status':'passed','model':config['model'],'results':results},indent=2,allow_nan=False)+'\n')
        save(stage='complete',results=results,max_memory_gb=torch.cuda.max_memory_allocated()/1e9)
    except BaseException:
        save(stage='failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
