"""Continue the frozen worker while preserving complete task prefixes."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from author_clone_bridge import ROOT, load_authors, verify_clone_sources, rudder_context, grains_context
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed, _validate_artifact, _versions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(args):
    output=args.output;state={'pid':os.getpid(),'worker':args.worker,'dtype':args.dtype,'completed_records':0,'started_at':time.time()}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/f'worker-{args.worker}.json',state)
    try:
        protocol=json.loads((output/'protocol.json').read_text())
        assert os.environ['CUDA_VISIBLE_DEVICES']==protocol['gpu_uuid']
        for f,h in protocol['source_hashes'].items():assert sha(ROOT/f)==h,f
        assert verify_clone_sources()==protocol['author_sources']
        config=json.loads((output/'config.json').read_text());config['model']['dtype']=args.dtype
        _seed(config);update(stage='loading_model');adapter=load_adapter(config['model']);authors=load_authors()
        artifacts={};vector=None
        if args.dtype=='bfloat16':
            for method in ['revis','vti']:
                artifact=torch.load(output/'artifacts'/f'{method}.pt',map_location='cpu',weights_only=True)
                assert sha(output/'artifacts'/f'{method}.pt')==protocol['retained_artifact_hashes'][method]
                _validate_artifact(artifact,config['model'],adapter=adapter)
                for group in ['vectors','vision_vectors']:
                    if group in artifact:artifact[group]={k:v.to(adapter.device) for k,v in artifact[group].items()}
                artifacts[method]=artifact
        else:
            seal=json.loads((output/'grains-artifact.json').read_text());assert seal['status']=='verified_complete'
            assert sha(output/'grains-author.pt')==seal['artifact_sha256']
            artifact=torch.load(output/'grains-author.pt',map_location='cpu',weights_only=True)
            assert artifact['model']==config['model']
            vector=artifact['vector'].to(adapter.device)
        datasets={b:read(output/'data'/f'{b}.jsonl') for b in ['pope','chair']}
        tasks=json.loads((output/'tasks.json').read_text())
        update(stage='ready',versions=_versions(),gpu_name=torch.cuda.get_device_name(0))
        for task in tasks:
            if task['dtype']!=args.dtype:continue
            claim=output/'claims'/task['id']
            if (claim/'complete.json').exists():continue
            try:(output/'continuation-claims'/task['id']).mkdir()
            except FileExistsError:continue
            claim.mkdir(exist_ok=True)
            rows=datasets[task['benchmark']][task['start']:task['stop']]
            path=output/'predictions'/f"{task['id']}.jsonl"
            prior_rows=read(path) if path.exists() else []
            assert len(prior_rows)<=len(rows)
            for prior,reference in zip(prior_rows,rows):
                assert all(prior.get(k)==v for k,v in reference.items())
                assert prior['method']==task['method'] and prior['dtype']==args.dtype
                assert prior['protocol_sha256']==protocol['sha256'] and prior['gpu_uuid']==protocol['gpu_uuid']
            save(claim/'owner.json',{'worker':args.worker,'pid':os.getpid(),'started_at':time.time(),'preserved_prefix_records':len(prior_rows)})
            update(stage='generating',task=task['id'],method=task['method'],benchmark=task['benchmark'],task_completed=len(prior_rows))
            with path.open('a' if path.exists() else 'x') as sink:
                for index,record in enumerate(rows[len(prior_rows):],start=len(prior_rows)):
                    key=(task['method'],record['id']);cached_path=output/'cache'/task['method']/f"{record['id'].replace(':','_')}.json"
                    if cached_path.exists():
                        cached=json.loads(cached_path.read_text())
                        assert cached['image_sha256']==record['image_sha256'] and cached['prompt']==record['prompt']
                        row={**cached,**record,'method':task['method'],'protocol_sha256':protocol['sha256'],
                             'dtype':args.dtype,'adopted_from_protocol':cached['protocol_sha256'],'adopted_cache_sha256':sha(cached_path)}
                    else:
                        assert sha(record['image'])==record['image_sha256'];_seed(config)
                        batch=adapter.prepare(record['image'],record['prompt']);diag={}
                        if task['method']=='rudder_clone':context=rudder_context(adapter,batch,task['benchmark'],authors,diag)
                        elif task['method']=='grains_clone':context=grains_context(adapter,vector,authors,diag)
                        elif task['method'] in ['revis','vti']:context=build_baseline(task['method'],adapter,artifacts[task['method']],config['methods'][task['method']])
                        else:context=contextlib.nullcontext()
                        torch.cuda.synchronize();start=time.perf_counter()
                        with context:response=adapter.generate(batch,max_new_tokens=record['max_new_tokens'])
                        torch.cuda.synchronize()
                        if task['method'] in ['rudder_clone','grains_clone']:assert diag['calls']>0
                        row={**record,'method':task['method'],'prediction':response,'generated_tokens':adapter.last_generated_tokens,
                             'elapsed_seconds':time.perf_counter()-start,'protocol_sha256':protocol['sha256'],
                             'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'worker':args.worker,'dtype':args.dtype,
                             'author_diagnostics':diag,'max_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1e9}
                        del batch
                    sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush()
                    update(completed_records=state['completed_records']+1,task_completed=index+1,
                           last_seconds=row.get('elapsed_seconds'),last_generated_tokens=row['generated_tokens'])
            save(claim/'complete.json',{'worker':args.worker,'records':len(rows),'sha256':sha(path),'completed_at':time.time()})
            print(json.dumps({'complete_task':task['id'],'records':len(rows)}),flush=True)
        assert verify_clone_sources()==protocol['author_sources']
        update(stage='complete',completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',required=True);p.add_argument('--dtype',choices=['float32','bfloat16'],required=True)
    main(p.parse_args())
