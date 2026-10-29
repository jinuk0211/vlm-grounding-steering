"""Extract fit50 using literal cloned GrAInS attribution, ablation and PCA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from author_clone_bridge import ROOT, GemmaAttributionAPI, load_authors, verify_clone_sources, extract_one
import numpy as np
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(output):
    status=output/'extraction-status.json';state={'pid':os.getpid(),'started_at':time.time(),'completed':0,'total':50}
    def update(**changes):state.update(changes,updated_at=time.time());save(status,state)
    try:
        protocol=json.loads((output/'protocol.json').read_text())
        assert os.environ['CUDA_VISIBLE_DEVICES']==protocol['gpu_uuid']
        for f,h in protocol['source_hashes'].items():assert sha(ROOT/f)==h,f
        assert verify_clone_sources()==protocol['author_sources']
        authors=load_authors();config=json.loads((output/'config.json').read_text());config['model']['dtype']='float32'
        _seed(config);update(stage='loading_fp32_model')
        adapter=load_adapter(config['model'])
        for p in adapter.model.parameters():p.requires_grad_(False)
        fit=[json.loads(s) for s in (output/'data/fit50.jsonl').read_text().splitlines()];assert len(fit)==50
        # Check that supplying token identities through Gemma's documented API
        # preserves its unscaled token-only forward before interpolating.
        batch=adapter.prepare(fit[0]['image'],fit[0]['prompt'],fit[0]['chosen'])
        ids=batch['input_ids'];mask=batch['attention_mask'];proxy=GemmaAttributionAPI(adapter.model)
        embed=proxy.get_input_embeddings();first=embed(ids);embed(ids)
        with torch.no_grad():
            native=adapter.model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,-1].float()
            supplied=proxy(inputs_embeds=first,attention_mask=mask).logits[:,-1].float()
        torch.testing.assert_close(native,supplied,rtol=1e-5,atol=1e-5)
        save(output/'gemma-ple-api-validation.json',{'status':'passed','max_logit_error':float((native-supplied).abs().max()),
             'scope':'Exact token-only forward at IG endpoint; token identity held fixed along IG; native context projection remains differentiable'})
        del native,supplied,first,batch,proxy,embed
        extraction=output/'extraction';extraction.mkdir()
        hidden={};original_ig=authors.grad.apply_integrated_gradients_contrastive
        with (output/'extraction.jsonl').open('x') as journal:
            for index,record in enumerate(fit):
                assert sha(record['image'])==record['metadata']['image_sha256']
                _seed(config);update(stage='extracting',index=index,record_id=record['id'])
                raw_path=extraction/f'{index:03d}.ig.pt'
                def observed(*args,**kwargs):
                    values=original_ig(*args,**kwargs)
                    assert all(torch.isfinite(v).all() for v in values)
                    torch.save({'positive':values[0].detach().cpu(),'negative':values[1].detach().cpu()},raw_path)
                    return values
                authors.grad.apply_integrated_gradients_contrastive=observed
                start=time.perf_counter();values=extract_one(adapter,record,authors)
                authors.grad.apply_integrated_gradients_contrastive=original_ig
                path=extraction/f'{index:03d}.npz';np.savez(path,**values)
                for name,value in values.items():
                    if name.startswith('hidden_'):hidden.setdefault(name,[]).append(value)
                entry={'index':index,'record':record,'artifact':str(path),'artifact_sha256':sha(path),
                       'raw_ig':str(raw_path),'raw_ig_sha256':sha(raw_path),'seconds':time.perf_counter()-start}
                journal.write(json.dumps(entry,allow_nan=False)+'\n');journal.flush()
                update(completed=index+1,last_seconds=entry['seconds'],max_memory_gb=torch.cuda.max_memory_allocated()/1e9)
                print(json.dumps({'extracted':index+1,'of':50,'seconds':entry['seconds']}),flush=True)
        arrays={k:np.stack(v) for k,v in hidden.items()};np.savez(output/'original-hidden-states.npz',**arrays)
        _seed(config);vector=authors.steering.compute_steering_vector(arrays,27,'pca')
        assert torch.isfinite(vector).all() and vector.norm()>0
        artifact={'vector':vector.cpu(),'layer':27,'alpha':10.0,'fit_records':[r['id'] for r in fit],
                  'model':config['model'],'author_commit':protocol['author_sources']['GrAInS']['commit'],
                  'ig_steps':20,'top_k':50,'mode':'pos','vector_method':'pca',
                  'architecture_bridge':'Gemma native PLE token identities fixed; native multimodal generation; author token-only IG/ablation retained'}
        torch.save(artifact,output/'grains-author.pt')
        assert verify_clone_sources()==protocol['author_sources']
        save(output/'grains-artifact.json',{'status':'verified_complete','artifact_sha256':sha(output/'grains-author.pt'),
             'hidden_states_sha256':sha(output/'original-hidden-states.npz'),'journal_sha256':sha(output/'extraction.jsonl'),
             'fit_manifest_sha256':sha(output/'data/fit50.jsonl'),'fit_records':50,'vector_norm':float(vector.norm()),
             'model':config['model'],'source_protocol_sha256':protocol['sha256']})
        update(stage='complete',completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
