"""Prepare author REVIS/VTI and run the user's complete fixed evaluation cohorts."""
import argparse,contextlib,csv,hashlib,importlib.util,json,math,os,subprocess,time,traceback
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed,_validate_artifact,_versions
from vlm_grounding.method import steer
from vlm_grounding.evaluation import _load_external_module,attach_chair_scores,evaluate_predictions
import author_revis_vti_bridge as author

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');DATA=ROOT/'data/hallucination-suite'
GPU='GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
METHODS=['base','ephs','revis','vti']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def helper():
    spec=importlib.util.spec_from_file_location('_gpu2_scoring_helper',ROOT/'scripts/audit_hallucination_suite.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def score(output,benchmark,protocol):
    h=helper();methods=['base','ephs'] if benchmark=='pope' else METHODS
    rows={m:read(output/'predictions'/f'{benchmark}-{m}.jsonl') for m in methods};n=3000 if benchmark=='pope' else 500
    metrics={};merged=output/'merged';merged.mkdir(exist_ok=True)
    assert all(len(v)==len({r['id'] for r in v})==n for v in rows.values())
    if benchmark=='chair':
        cache=json.loads((DATA/'chair-scorer-state.json').read_text());source=ROOT/'upstream/chair/chair.py';assert sha(source)==cache['scorer_sha256']
        assert cache['built_with_unmodified_constructor']
        for name,digest in cache['annotation_hashes'].items():assert sha(DATA/'annotations'/name)==digest
        original=protocol['original_data_protocol'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source.parent,text=True).strip()==original['chair_scoring']['port_commit']
        subprocess.run(['git','diff','--exit-code','HEAD','--','chair.py'],cwd=source.parent,check=True,capture_output=True)
        module=_load_external_module(source,'_gpu2_official_chair');scorer=module.CHAIR.__new__(module.CHAIR);scorer.__dict__.update(cache['state']);scorer.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    for method in methods:
        values=rows[method]
        if benchmark=='pope':
            metrics[method]=h.pope_metrics(values);h.verify_upstream_pope(values,output,method,'adversarial',metrics[method]);metrics[method]['strict']=evaluate_predictions(values)
        else:
            captions=merged/f'{method}.chair.captions.json';save(captions,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in values]);official=scorer.compute_chair(str(captions),'image_id','caption');save(merged/f'{method}.chair.official.json',official);values=attach_chair_scores(values,official);rows[method]=values;metrics[method]=evaluate_predictions(values)
            assert math.isclose(metrics[method]['CHAIR_s'],official['overall_metrics']['CHAIRs']) and math.isclose(metrics[method]['CHAIR_i'],official['overall_metrics']['CHAIRi'])
        metrics[method]['at_token_cap']=sum(r['generated_tokens']>=r['max_new_tokens'] for r in values);metrics[method]['generated_tokens_mean']=sum(r['generated_tokens'] for r in values)/n
        h.write_jsonl(merged/f'{benchmark}-{method}.jsonl',values)
    report={'status':'scored_complete','benchmark':benchmark,'cases_per_method':n,'methods':methods,'metrics':metrics,'uncertainty':h.intervals(rows,benchmark),'protocol':protocol,'total_responses':n*len(methods),'official_scorer_used':True}
    save(output/f'{benchmark}-report.json',report)
    columns=['method']+list(dict.fromkeys(k for value in metrics.values() for k,v in value.items() if not isinstance(v,dict)))
    with (output/f'{benchmark}-metrics.csv').open('w',newline='') as sink:
        writer=csv.DictWriter(sink,fieldnames=columns);writer.writeheader()
        for method,value in metrics.items():writer.writerow({'method':method,**{k:v for k,v in value.items() if not isinstance(v,dict)}})
    return report

def main(args):
    output=Path(args.output);status=output/'evaluation-status.json';state={'pid':os.getpid(),'started_at':time.time(),'gpu_uuid':GPU,'completed_records':0}
    def update(**value):state.update(value,updated_at=time.time());save(status,state)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and torch.cuda.device_count()==1
        config=json.loads((ROOT/'configs/gemma4_e4b.local.json').read_text());_seed(config);update(stage='loading_model');adapter=load_adapter(config['model']);adapter.model.config._commit_hash=config['model']['revision']
        ephs=json.loads((output/'ephs.json').read_text());_validate_artifact(ephs,config['model'],adapter=adapter)
        assert json.loads((output/'ephs-validation.json').read_text())['status']=='passed'
        if not (output/'author-revis.pt').exists() or not (output/'author-vti.pt').exists():
            revis,vti=author.extract(adapter,output,update)
        else:
            revis=torch.load(output/'author-revis.pt',map_location='cpu',weights_only=True);vti=torch.load(output/'author-vti.pt',map_location='cpu',weights_only=True)
        # These are the actual pinned author module bodies, restored to Gemma's dtype.
        artifacts={'base':None,'ephs':ephs,'revis':revis,'vti':vti}
        def context(method,diagnostics=None):
            if method=='base':return contextlib.nullcontext()
            if method=='ephs':return steer(adapter,ephs,config['methods']['ephs'])
            if method=='revis':return author.revis_context(adapter,revis,diagnostics)
            return author.vti_context(adapter,vti,diagnostics)
        update(stage='checking_real_model_hooks');development=read(output/'data/author-preferences.jsonl')[0];checks={}
        for method in METHODS:
            batch=adapter.prepare(development['image_path'],'Please describe this image in detail.');diagnostics={}
            with context(method,diagnostics):answer=adapter.generate(batch,max_new_tokens=32)
            assert answer.strip() and adapter.last_generated_tokens>0
            checks[method]={'response':answer,'tokens':adapter.last_generated_tokens,'diagnostics':diagnostics}
        assert checks['vti']['diagnostics'].get('changed_elements',0)>0
        calibration=read(output/'author-revis-calibration.jsonl');probe=max(calibration,key=lambda r:r['risks'][str(revis['layer'])]);diagnostics={}
        with context('revis',diagnostics):answer=adapter.generate(adapter.prepare(probe['image'],probe['prompt']),max_new_tokens=5)
        assert diagnostics.get('changed_elements',0)>0,'REVIS hook never activates even on high-risk calibration input'
        checks['revis_gate_probe']={'response':answer,'diagnostics':diagnostics}
        fit_probe=read(output/'data/fit.jsonl')[0];batch=adapter.prepare(fit_probe['image'],fit_probe['prompt'])
        with torch.no_grad():base_logits=adapter.forward(batch).logits[0,-1].float()
        with context('ephs'),torch.no_grad():changed_logits=adapter.forward(batch).logits[0,-1].float()
        delta=float((base_logits-changed_logits).abs().max());assert math.isfinite(delta) and delta>0
        checks['ephs_first_token_max_logit_delta']=delta
        save(output/'actual-model-preflight.json',{'status':'passed','checks':checks,'gpu_uuid':GPU,'versions':_versions()})
        sourcefiles=['src/vlm_grounding/backend.py','src/vlm_grounding/method.py','src/vlm_grounding/cli.py','src/vlm_grounding/evaluation.py','scripts/author_revis_vti_bridge.py','scripts/run_gpu2_requested_eval.py','scripts/audit_hallucination_suite.py']
        for name,module in [('backend.py',__import__('vlm_grounding.backend',fromlist=[''])),('method.py',__import__('vlm_grounding.method',fromlist=['']))]:assert sha(module.__file__)==sha(ROOT/'src/vlm_grounding'/name)
        previous=Path(json.loads((ROOT/'outputs/pope-selected-current.json').read_text())['output']);original=json.loads((previous/'protocol.json').read_text())
        protocol={'model':config,'gpu_index':2,'gpu_uuid':GPU,'automatic_other_gpu_fallback':False,'methods':{'pope':['base','ephs'],'chair':METHODS},'cohorts':{'pope_adversarial':3000,'chair_coco_val2014':500},'generation':{'seed_per_example':42,'do_sample':False,'pope_max_new_tokens':128,'chair_max_new_tokens':512},'source_hashes':{n:sha(ROOT/n) for n in sourcefiles},'artifact_hashes':{n:sha(output/n) for n in ['ephs.json','author-revis.pt','author-vti.pt']},'manifest_hashes':{n:sha(DATA/n) for n in ['pope.jsonl','chair.jsonl']},'author_sources':author.sources(),'author_development':json.loads((output/'author-development.json').read_text()),'ephs_training_protocol':json.loads((output/'training-protocol.json').read_text()),'original_data_protocol':original,'implementation':{'ephs':'native proposed EPHS','revis':'pinned author PCA, calibration equations and all-position hook; Gemma I/O bridge','vti':'pinned author PCA and MLP hook; Gemma patch layout and BF16 dtype bridge'},'author_reproduction':False}
        protocol['author_manifest_hashes']={n:sha(output/'data'/n) for n in ['author-preferences.jsonl','author-calibration.jsonl']}
        protocol['author_image_hashes']={r['image_path']:r['image_sha256'] for r in read(output/'data/author-preferences.jsonl')}
        protocol['chair_cache_sha256']=sha(DATA/'chair-scorer-state.json')
        protocol['gemma_io_bridge']={'fixed_input_image_size':448,'vision_patch_size':16,'vision_soft_token_limit':280,'vision_padding_preserved':True,'vti_caption_state':'last caption token without appended EOS','revis_caption_state':'author full dialogue template','vti_return_dtype':'author FP16 result explicitly cast back to Gemma BF16'}
        protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (output/'evaluation-protocol.json').exists():assert json.loads((output/'evaluation-protocol.json').read_text())==protocol
        else:save(output/'evaluation-protocol.json',protocol)
        if args.prepare_only:update(stage='prepared',completed_at=time.time());return
        (output/'predictions').mkdir(exist_ok=True)
        manifests={'pope':[r for r in read(DATA/'pope.jsonl') if r['subset']=='adversarial'],'chair':read(DATA/'chair.jsonl')}
        devhash={sha(r['image']) for r in read(output/'data/discovery.jsonl')+read(output/'data/fit.jsonl')}|{r['image_sha256'] for r in read(output/'data/author-preferences.jsonl')}
        for benchmark,records in manifests.items():
            assert len(records)==(3000 if benchmark=='pope' else 500)
            assert not devhash.intersection(r['image_sha256'] for r in records)
            methods=protocol['methods'][benchmark];sinks={};completed={}
            for method in methods:
                path=output/'predictions'/f'{benchmark}-{method}.jsonl';existing=read(path) if path.exists() else []
                for row in existing:assert row['protocol_sha256']==protocol['sha256']
                completed[method]={r['id']:r for r in existing};assert len(completed[method])==len(existing)
                sinks[method]=path.open('a')
            try:
                for index,record in enumerate(records):
                    assert sha(record['image'])==record['image_sha256']
                    cap=128 if benchmark=='pope' else 512
                    for method in methods:
                        if record['id'] in completed[method]:continue
                        _seed(config);update(stage='generating',benchmark=benchmark,method=method,image_index=index,benchmark_completed=sum(len(v) for v in completed.values()),benchmark_expected=len(records)*len(methods))
                        batch=adapter.prepare(record['image'],record['prompt']);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                        with context(method):prediction=adapter.generate(batch,max_new_tokens=cap)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        row={**record,'method':method,'prediction':prediction,'max_new_tokens':cap,'generated_tokens':adapter.last_generated_tokens,'elapsed_seconds':elapsed,'protocol_sha256':protocol['sha256'],'gpu_uuid':GPU,'max_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1e9}
                        sinks[method].write(json.dumps(row,allow_nan=False)+'\n');sinks[method].flush();completed[method][row['id']]=row
                        state['completed_records']+=1;update(last_seconds=elapsed,last_generated_tokens=row['generated_tokens'],benchmark_completed=sum(len(v) for v in completed.values()))
            finally:
                for sink in sinks.values():sink.close()
            update(stage='scoring',benchmark=benchmark);score(output,benchmark,protocol)
        for name,digest in protocol['source_hashes'].items():assert sha(ROOT/name)==digest
        update(stage='complete',completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare-only',action='store_true');main(p.parse_args())
